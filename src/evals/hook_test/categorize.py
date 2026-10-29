"""Classify exported videos into content categories from title keywords.

The channel's upload history mixes Brawl Stars / Clash Royale with dozens of
hyper-casual mobile games and some non-gaming posts. Game type is a selection
confound for the hook test (see plan/02-hook-test.md section 2), so every video
gets a ``category`` column that ``select.py`` can filter on.

Writes ``data/evals/hook_test/categories.csv``. Deterministic, no network.

    ./venv/bin/python -m src.evals.hook_test.categorize [--export DIR]
"""

from __future__ import annotations

import re
from pathlib import Path

# First match wins. Keywords are lowercase regex alternations matched against
# the lowercased title; keep them specific enough to avoid cross-talk
# (e.g. "run"/"game" alone would match half the corpus, so they only count
# inside the game-pattern group after the supercell group).
CATEGORY_PATTERNS: list[tuple[str, str]] = [
    ("supercell", r"brawl|clash"),
    (
        "misc_mobile_games",
        r"dop|delete one|gameplay|walkthrough|level ?\d|puzzle|simulator|race|"
        r"runner|running|stack|dash|angela|kitty|sonic|spider|skibidi|sakura|"
        r"zombie|barbie|minecraft|thief|finger|sandwich|juice|teeth|tall man|"
        r"rope man|donut|bike|bicycle|outfit|queen|squeeze|ice ?cream|"
        r"shape shifting|build a|talking",
    ),
]


def categorize_title(title: str) -> str:
    """Map one video title to a category; unmatched titles are 'non_gaming_other'."""
    t = (title or "").lower()
    for category, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, t):
            return category
    return "non_gaming_other"


def latest_export(base: Path) -> Path:
    """Newest timestamped export directory under base."""
    dirs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "videos.csv").exists())
    if not dirs:
        raise FileNotFoundError(f"no export with videos.csv under {base}")
    return dirs[-1]
