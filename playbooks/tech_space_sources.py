"""Conectores abiertos de espacio y tecnología (0 tokens LLM, sin API key).

Fuentes verificadas en vivo:
- The Space Devs — Launch Library 2 API: lanzamientos globales de SpaceX, Blue Origin,
  Virgin Galactic, Japón (JAXA), India (ISRO) y China (CASC/CNSA/LandSpace).
- GitHub Public REST API: actividad, descripción y últimos commits de repos `owner/repo`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import requests

from playbooks.open_sources import DEFAULT_HEADERS, HTTP_TIMEOUT_SECONDS

LAUNCH_LIBRARY_PREV_URL = "https://ll.thespacedevs.com/2.2.0/launch/previous/"
LAUNCH_LIBRARY_NEXT_URL = "https://ll.thespacedevs.com/2.2.0/launch/upcoming/"
GITHUB_REPO_URL = "https://api.github.com/repos/{slug}"
GITHUB_COMMITS_URL = "https://api.github.com/repos/{slug}/commits"

_SPACE_PROVIDERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("spacex", "falcon 9", "falcon heavy", "starship", "dragon"), "SpaceX"),
    (("blue origin", "new glenn", "new shepard"), "Blue Origin"),
    (("virgin galactic", "virgin orbit", "spaceshiptwo"), "Virgin"),
    (("jaxa", "japan", "h3 rocket", "h-iia", "epsilon"), "JAXA"),
    (("isro", "india", "gaganyaan", "pslv", "gslv", "lvm3"), "ISRO"),
    (("china", "cnsa", "casc", "long march", "landspace", "zhuque", "tiangong"), "China"),
)

_GITHUB_SLUG_RE = re.compile(r"\b([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)\b")
_NON_REPO_SLUGS = frozenset({"y/y", "q/q", "m/m", "mb/d", "o/u", "u/s", "n/a", "w/l"})


@dataclass(frozen=True)
class LaunchRecord:
    """Resumen de un lanzamiento orbital desde Launch Library 2."""

    date: str
    name: str
    provider: str
    status: str


@dataclass(frozen=True)
class GitHubRepoSummary:
    """Estado reciente de un repositorio público de GitHub."""

    slug: str
    stars: int
    pushed_at: str
    description: str
    recent_commits: tuple[str, ...]


def detect_space_provider(question_text: str) -> str | None:
    """Identifica el proveedor o agencia espacial clave en la pregunta."""
    lower = question_text.lower()
    for keywords, provider in _SPACE_PROVIDERS:
        if any(kw in lower for kw in keywords):
            return provider
    return None


def _get_json_with_retry(
    client: requests.Session,
    url: str,
    params: dict[str, str | int],
) -> dict | list | None:
    for _ in range(2):
        try:
            resp = client.get(url, params=params, headers=DEFAULT_HEADERS, timeout=HTTP_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return resp.json()
            return None
        except (requests.RequestException, ValueError):
            continue
    return None


def fetch_launch_library_summary(
    question_text: str,
    *,
    session: requests.Session | None = None,
    limit: int = 5,
) -> tuple[list[LaunchRecord], list[LaunchRecord]]:
    """Obtiene últimos lanzamientos y próximos agendados en Launch Library 2."""
    provider = detect_space_provider(question_text)
    if provider is None:
        return [], []

    client = session or requests.Session()
    params_prev: dict[str, str | int] = {"search": provider, "limit": limit}
    params_next: dict[str, str | int] = {"search": provider, "limit": 3}

    prev_payload = _get_json_with_retry(client, LAUNCH_LIBRARY_PREV_URL, params_prev)
    next_payload = _get_json_with_retry(client, LAUNCH_LIBRARY_NEXT_URL, params_next)

    def _parse(payload: dict | list | None) -> list[LaunchRecord]:
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            return []
        parsed: list[LaunchRecord] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            date_s = str(item.get("net") or "")[:10] or "unknown"
            name = str(item.get("name") or "").strip()
            lsp = str((item.get("launch_service_provider") or {}).get("name") or provider)
            status = str((item.get("status") or {}).get("abbrev") or "unknown")
            if name:
                parsed.append(LaunchRecord(date=date_s, name=name[:100], provider=lsp, status=status))
        return parsed

    return _parse(prev_payload), _parse(next_payload)


def extract_github_repos(question_text: str) -> list[str]:
    """Extrae slugs `owner/repo` válidos mencionados en la pregunta."""
    found: list[str] = []
    for match in _GITHUB_SLUG_RE.findall(question_text):
        low = match.lower()
        if low in _NON_REPO_SLUGS or match.startswith(("http", "www")):
            continue
        owner, repo = match.split("/", 1)
        if len(owner) < 2 or len(repo) < 2 or owner.isdigit() or repo.isdigit():
            continue
        if match not in found:
            found.append(match)
    return found[:2]


def fetch_github_repo_summary(
    slug: str,
    *,
    session: requests.Session | None = None,
) -> GitHubRepoSummary | None:
    """Consulta la API pública de GitHub para obtener pulso y commits recientes."""
    client = session or requests.Session()
    headers = {**DEFAULT_HEADERS, "Accept": "application/vnd.github+json"}
    try:
        resp = client.get(
            GITHUB_REPO_URL.format(slug=slug),
            headers=headers,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if resp.status_code != 200:
            return None
        repo_data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    if not isinstance(repo_data, dict):
        return None

    commits: list[str] = []
    try:
        c_resp = client.get(
            GITHUB_COMMITS_URL.format(slug=slug),
            params={"per_page": 3},
            headers=headers,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if c_resp.status_code == 200 and isinstance(c_resp.json(), list):
            for c in c_resp.json()[:3]:
                commit_obj = c.get("commit") if isinstance(c, dict) else None
                if not isinstance(commit_obj, dict):
                    continue
                msg = str(commit_obj.get("message") or "").splitlines()[0][:100]
                c_date = str((commit_obj.get("author") or {}).get("date") or "")[:10]
                if msg:
                    commits.append(f"[{c_date}] {msg}")
    except (requests.RequestException, ValueError):
        pass

    return GitHubRepoSummary(
        slug=slug,
        stars=int(repo_data.get("stargazers_count") or 0),
        pushed_at=str(repo_data.get("pushed_at") or "")[:10],
        description=str(repo_data.get("description") or "").strip()[:140],
        recent_commits=tuple(commits),
    )
