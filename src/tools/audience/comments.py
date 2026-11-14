"""Own-channel comments as evidence for the audience reaction (plan/09 step R1).

    ./venv/bin/python -m src.tools.audience.comments          # the free E1 test

Top-level comments only; the creator's own comments are dropped; author names and ids never leave this module.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.evals.hook_test import config as C

MENTION = re.compile(r"@[\w.\-]+")


def export_dir() -> Path:
    return Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"])


def clean_comments(comments: pd.DataFrame, own_channel_id: str) -> pd.DataFrame:
    """video_id, text, like_count for top-level comments by viewers, most liked first; @mentions removed."""
    c = comments[~comments["is_reply"].astype(bool) & (comments["author_channel_id"] != own_channel_id)]
    text = c["text_original"].astype(str).map(lambda t: MENTION.sub("", t).strip())
    out = pd.DataFrame({"video_id": c["video_id"].to_numpy(), "text": text.to_numpy(),
                        "like_count": c["like_count"].fillna(0).astype(int).to_numpy()})
    out = out[out["text"].str.len() > 0].drop_duplicates(["video_id", "text"])
    return out.sort_values(["video_id", "like_count"], ascending=[True, False], kind="stable").reset_index(drop=True)


def load_comments() -> pd.DataFrame:
    d = export_dir()
    return clean_comments(pd.read_csv(d / "comments.csv"), pd.read_csv(d / "channel.csv")["id"].iloc[0])


def top_comments(comments: pd.DataFrame, video_id: str, n: int = 8, max_chars: int = 200) -> list[dict]:
    c = comments[comments["video_id"] == video_id].head(n)
    return [{"text": t[:max_chars], "likes": int(l)} for t, l in zip(c["text"], c["like_count"])]
