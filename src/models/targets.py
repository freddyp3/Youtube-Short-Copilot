"""Other targets and drop-off timing on train + DEV (plan/03c section 14). GATE/HELDOUT are never loaded.

    ./venv/bin/python -m src.models.targets run        # nested CV, 8 targets x 4 recipes
    ./venv/bin/python -m src.models.targets perm       # one permutation test over all 32 pairs
    ./venv/bin/python -m src.models.targets dropoff    # do extracted timestamps mark where viewers leave
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evals.hook_test.score import criterion_b, ground_truth_for, residualize

CURVE_TARGETS = ("early_hold", "mid_hold", "late_hold", "replay_level", "mean_rrp")
RATE_TARGETS = ("like_rate", "share_rate", "comment_rate")
TARGETS = CURVE_TARGETS + RATE_TARGETS


def curve_targets(retention: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vid, g in retention.sort_values("elapsedVideoTimeRatio").groupby("video_id"):
        x, y = g["elapsedVideoTimeRatio"].to_numpy(), g["audienceWatchRatio"].to_numpy()
        w = lambda r: float(np.interp(r, x, y))
        rows.append({"video_id": vid, "early_hold": w(0.25) / w(0.02), "mid_hold": w(0.75) / w(0.25),
                     "late_hold": w(0.95) / w(0.75), "replay_level": w(0.02),
                     "mean_rrp": float(g["relativeRetentionPerformance"].mean())})
    return pd.DataFrame(rows)


def rate_targets(metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics[metrics["creatorContentType"] == "shorts"]
    out = m[["video_id"]].copy()
    for name, col in zip(RATE_TARGETS, ("likes", "shares", "comments")):
        out[name] = np.log((m[col] + 1) / m["engagedViews"])
    return out


def residual_on_views(y: np.ndarray, views: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(y)), np.log(views)])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def add_targets(df: pd.DataFrame, export) -> pd.DataFrame:
    """Raw targets joined on video_id, then residualised on log(duration) and era using these rows only."""
    df = df.drop(columns=["mean_rrp"], errors="ignore")  # same definition, recomputed below
    df = df.merge(curve_targets(export.retention), on="video_id", how="left", validate="one_to_one")
    df = df.merge(rate_targets(export.metrics), on="video_id", how="left", validate="one_to_one")
    if df[list(TARGETS)].isna().any().any():
        raise ValueError("some videos have no retention curve or metrics row")
    for t in TARGETS:
        df[f"{t}_raw"] = df[t]
        df[t] = residualize(df[t], df["duration_seconds"], df["published_at"]).to_numpy()
        df[f"{t}_views"] = residual_on_views(df[t].to_numpy(), df["engagedViews"].to_numpy(dtype=float))
    return df
