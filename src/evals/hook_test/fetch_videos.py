"""Map Google Takeout video files to video_ids, hardlink them into data/videos/, and blind the selection.

Takeout names files by title (filesystem-illegal chars become ``_``, duplicate titles get ``(N)``
suffixes), so matching is: exact normalized title -> fuzzy title -> duration disambiguation for
ties -> ``--resolve`` manual overrides. Writes ``data/evals/hook_test/video_map.csv``, copies the
raw Takeout metadata into ``data/channel/takeout/``, and produces metadata-stripped blind copies
plus contact sheets for the leakage eyeball check (plan/02-hook-test.md section 3).

    ./venv/bin/python -m src.evals.hook_test.fetch_videos \\
        --src "~/Downloads/Takeout/YouTube and YouTube Music" [--export DIR] \\
        [--resolve VIDEO_ID=PATH ...] [--no-blind]
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Chars Takeout turns into "_" in filenames: filesystem-illegal ones plus quotes/apostrophes.
TAKEN_OUT_CHARS = set('\\/:*?"<>|\'’‘“”')
DUP_SUFFIX_RE = re.compile(r"\(\d+\)\s*$")
FUZZY_MIN_RATIO = 0.9
DURATION_TOLERANCE_S = 1.0


# ---------------------------------------------------------------- pure helpers

def normalize_exact(s: str) -> str:
    """Title <-> filename-stem normalization for exact matching.

    NFC, strip a trailing ``(N)`` dup suffix, strip whitespace, map Takeout's
    replaced chars to ``_``.
    """
    s = unicodedata.normalize("NFC", s)
    s = DUP_SUFFIX_RE.sub("", s)
    s = s.strip()
    s = "".join("_" if c in TAKEN_OUT_CHARS else c for c in s)
    # Takeout collapses runs of replaced chars: "game??!?" -> "game_!_"
    return re.sub(r"_+", "_", s)


def normalize_fuzzy(s: str) -> str:
    """Aggressive normalization for fuzzy matching: alnum only, ``_``-joined, casefolded."""
    s = normalize_exact(s).casefold()
    s = "".join(c if c.isalnum() else "_" for c in s)
    return re.sub(r"_+", "_", s).strip("_")


def parse_resolve(pairs: list[str]) -> dict[str, str]:
    """Parse ``VIDEO_ID=PATH`` overrides into {video_id: path}."""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--resolve expects VIDEO_ID=PATH, got {p!r}")
        vid, path = p.split("=", 1)
        out[vid.strip()] = path.strip()
    return out


def load_takeout_metadata(meta_dir: Path) -> pd.DataFrame:
    """Concat the ``videos*.csv`` chunks (differing column sets) into one df keyed by Video ID."""
    files = sorted(meta_dir.glob("videos*.csv"))
    if not files:
        raise FileNotFoundError(f"no videos*.csv under {meta_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.drop_duplicates(subset="Video ID").set_index("Video ID")
    log.info("takeout metadata: %d unique video ids from %s", len(df), meta_dir)
    return df


def merged_takeout_videos(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per video_id with the column set the repo keeps."""
    tag_cols = [c for c in meta.columns if re.fullmatch(r"Tag \d+", c)]
    tags = meta[tag_cols].apply(
        lambda row: "|".join(str(v) for v in row if pd.notna(v) and str(v).strip()), axis=1
    )
    out = pd.DataFrame({
        "video_id": meta.index,
        "title": meta["Video Title (Original)"],
        "description": meta.get("Video Description (Original)"),
        "privacy": meta["Privacy"],
        "state": meta["Video State"],
        "category": meta["Video Category"],
        "channel_id": meta.get("Channel ID"),
        "approx_duration_s": pd.to_numeric(meta["Approx Duration (ms)"], errors="coerce") / 1000.0,
        "created_at": meta["Video Create Timestamp"],
        "published_at": meta["Video Publish Timestamp"],
        "tags": tags,
    })
    return out.reset_index(drop=True)


@dataclass
class FileEntry:
    path: Path
    stem: str
    duration: float | None


@dataclass
class Match:
    video_id: str
    method: str


def match_files(
    files: list[FileEntry],
    meta: pd.DataFrame,
    overrides: dict[str, str],
) -> dict[Path, Match]:
    """Assign each file a video_id. Every id and every file used at most once."""
    matches: dict[Path, Match] = {}
    used_ids: set[str] = set()

    # --resolve overrides first (paths are as passed; resolve against cwd).
    by_path = {f.path.resolve(): f for f in files}
    for vid, raw in overrides.items():
        entry = by_path.get(Path(raw).expanduser().resolve())
        if entry is None:
            log.warning("--resolve %s=%s: path not found among scanned files", vid, raw)
            continue
        if vid in used_ids:
            log.warning("--resolve %s: video_id already used, skipping", vid)
            continue
        matches[entry.path] = Match(vid, "manual")
        used_ids.add(vid)

    remaining = [f for f in files if f.path not in matches]
    ids = [vid for vid in meta.index if vid not in used_ids]

    exact_ids: dict[str, list[str]] = {}
    for vid in ids:
        exact_ids.setdefault(normalize_exact(str(meta.at[vid, "Video Title (Original)"])), []).append(vid)
    exact_files: dict[str, list[FileEntry]] = {}
    for f in remaining:
        exact_files.setdefault(normalize_exact(f.stem), []).append(f)

    def take(entry: FileEntry, vid: str, method: str) -> None:
        matches[entry.path] = Match(vid, method)
        used_ids.add(vid)

    # title_exact: key with exactly one unmatched id and one file
    for key, fs in exact_files.items():
        cand_ids = [v for v in exact_ids.get(key, []) if v not in used_ids]
        free = [f for f in fs if f.path not in matches]
        if len(cand_ids) == 1 and len(free) == 1:
            take(free[0], cand_ids[0], "title_exact")

    # title_fuzzy: unique best difflib ratio >= threshold over remaining ids
    fuzzy_ids = {v: normalize_fuzzy(str(meta.at[v, "Video Title (Original)"]))
                 for v in ids if v not in used_ids}
    for f in remaining:
        if f.path in matches:
            continue
        target = normalize_fuzzy(f.stem)
        scored = sorted(
            ((difflib.SequenceMatcher(None, target, fkey).ratio(), v)
             for v, fkey in fuzzy_ids.items() if v not in used_ids),
            reverse=True,
        )
        if not scored or scored[0][0] < FUZZY_MIN_RATIO:
            continue
        if len(scored) > 1 and scored[1][0] == scored[0][0]:
            log.info("fuzzy tie for %s (%s vs %s), leaving for duration step",
                     f.path.name, scored[0][1], scored[1][1])
            continue
        take(f, scored[0][1], "title_fuzzy")

    # title+duration: disambiguate remaining ties within shared exact keys.
    # A file only claims an id when its best diff is strictly better than the
    # runner-up; contested ids go to the smaller diff and the loser re-evaluates.
    for key, fs in exact_files.items():
        for _ in range(len(fs) + 1):
            cand_ids = [v for v in exact_ids.get(key, []) if v not in used_ids]
            free = [f for f in fs if f.path not in matches]
            if not cand_ids or not free:
                break
            claims: dict[str, tuple[float, FileEntry]] = {}
            for f in free:
                if f.duration is None:
                    continue
                diffs = sorted(
                    (abs(f.duration - float(meta.at[v, "Approx Duration (ms)"]) / 1000.0), v)
                    for v in cand_ids
                    if pd.notna(pd.to_numeric(meta.at[v, "Approx Duration (ms)"], errors="coerce"))
                )
                diffs = [d for d in diffs if d[0] <= DURATION_TOLERANCE_S]
                if not diffs or (len(diffs) > 1 and diffs[1][0] == diffs[0][0]):
                    continue
                diff, vid = diffs[0]
                if vid not in claims or diff < claims[vid][0]:
                    claims[vid] = (diff, f)
            if not claims:
                break
            for vid, (_, f) in claims.items():
                if f.path not in matches and vid not in used_ids:
                    take(f, vid, "title+duration")

    return matches
