"""Update README traffic totals from GitHub traffic APIs.

GitHub exposes traffic history for a recent window only. This script stores
daily traffic rows in .github/traffic-history.json and computes cumulative
display totals from the saved history.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
HISTORY = ROOT / ".github" / "traffic-history.json"
TRAFFIC_START = "<!-- traffic:start -->"
TRAFFIC_END = "<!-- traffic:end -->"
DEFAULT_REPOSITORY = "XinzeLee/PE-GPT"
MONITORING_STARTED = "May, 23, 2026"


def api_get(path: str) -> dict:
    token = os.environ.get("TRAFFIC_STATS_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "TRAFFIC_STATS_TOKEN, GH_TOKEN, or GITHUB_TOKEN is required to read GitHub traffic data. "
            "For GitHub Actions, create a repository secret named TRAFFIC_STATS_TOKEN."
        )

    repository = os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY)
    url = f"https://api.github.com/repos/{repository}{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "traffic-stats-updater",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403:
            raise RuntimeError(
                "GitHub traffic API returned 403 Forbidden. The default GITHUB_TOKEN usually cannot read "
                "repository traffic metrics. Create a repository secret named TRAFFIC_STATS_TOKEN using a "
                "classic PAT with repo scope, or a fine-grained PAT for XinzeLee/PE-GPT with read access to "
                f"repository Administration/traffic metrics, then rerun the workflow. API response: {detail}"
            ) from exc
        raise RuntimeError(f"GitHub API request failed: {exc.code} {detail}") from exc


def load_history() -> dict:
    if not HISTORY.exists():
        return {"views": {}, "clones": {}}
    return json.loads(HISTORY.read_text(encoding="utf-8"))


def merge_daily_rows(history: dict, key: str, rows: list[dict]) -> None:
    bucket = history.setdefault(key, {})
    for row in rows:
        date = str(row["timestamp"])[:10]
        bucket[date] = {
            "count": int(row.get("count", 0)),
            "uniques": int(row.get("uniques", 0)),
        }


def totals(history: dict, key: str) -> tuple[int, int]:
    rows = history.get(key, {}).values()
    return (
        sum(int(row.get("count", 0)) for row in rows),
        sum(int(row.get("uniques", 0)) for row in rows),
    )


def fmt(n: int) -> str:
    return f"{n:,}"


def badge_value(n: int) -> str:
    return fmt(n).replace("-", "--").replace("_", "__").replace(" ", "_")


def render_block(history: dict) -> str:
    views, _unique_visitors = totals(history, "views")
    clones, unique_clones = totals(history, "clones")
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return f"""{TRAFFIC_START}
<p align="center">
  <a href="https://github.com/XinzeLee/PE-GPT/graphs/traffic">
    <img src="https://img.shields.io/badge/Total_Views-{badge_value(views)}-2563eb?style=flat-square" alt="Total repository views: {fmt(views)}" />
  </a>
  <a href="https://github.com/XinzeLee/PE-GPT/graphs/traffic">
    <img src="https://img.shields.io/badge/Total_Clones-{badge_value(clones)}-7c3aed?style=flat-square" alt="Total repository clones: {fmt(clones)}" />
  </a>
  <a href="https://github.com/XinzeLee/PE-GPT/graphs/traffic">
    <img src="https://img.shields.io/badge/Unique_Clones-{badge_value(unique_clones)}-b45309?style=flat-square" alt="Unique repository clones: {fmt(unique_clones)}" />
  </a>
</p>

<p align="center"><sub>Github traffic (monitoring started on {MONITORING_STARTED}) · cumulative tracked totals · Till {updated} UTC</sub></p>
{TRAFFIC_END}"""


def update_readme(history: dict) -> None:
    readme = README.read_text(encoding="utf-8")
    block = render_block(history)
    pattern = re.compile(
        rf"{re.escape(TRAFFIC_START)}.*?{re.escape(TRAFFIC_END)}",
        flags=re.DOTALL,
    )
    if pattern.search(readme):
        readme = pattern.sub(block, readme, count=1)
    else:
        readme = readme.replace("## Description\n", f"{block}\n\n## Description\n", 1)
    README.write_text(readme, encoding="utf-8", newline="\n")


def main() -> int:
    history = load_history()
    views = api_get("/traffic/views")
    clones = api_get("/traffic/clones")
    merge_daily_rows(history, "views", views.get("views", []))
    merge_daily_rows(history, "clones", clones.get("clones", []))

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_readme(history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
