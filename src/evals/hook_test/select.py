"""Pick the strong/weak Shorts for the hook test and split them into DEV / GATE / HELDOUT (plan section 2).

Pipeline: per-video features -> eligible pool (funnel) -> hold_resid label -> strong/weak
candidates (top/bottom quartile) -> cell-matched strong/weak pairs -> splits -> balance check
(re-draw with seed+1 on failure). Deterministic, no network. Writes selection.json + selection.csv.

Deviation from the plan: no relativeRetentionPerformance sanity filter. On this channel the pool
median RRP is 0.65 and RRP correlates -0.67 with duration, so the filter removed half the weak
candidates and reintroduced the length confound. mean_rrp is kept as a secondary label.

    ./venv/bin/python -m src.evals.hook_test.select [--export DIR] [--available video_map.csv]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.categorize import categorize_video, hold_resid
from src.evals.hook_test.data import ExportTables, load_export

class SelectionError(RuntimeError):
    """Raised when the pool is too small or no seed yields a balanced draw."""


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def build_frame(t: ExportTables) -> pd.DataFrame:
    """One row per Short with every column the funnel and label need."""
    v = t.videos[t.videos["content_type"] == "shorts"][
        ["video_id", "title", "published_at", "duration_seconds"]
    ].copy()
    v["category"] = [categorize_video(i, t) for i, t in zip(v["video_id"], v["title"])]
    m = t.metrics[t.metrics["creatorContentType"] == "shorts"][
        ["video_id", "engagedViews", "averageViewPercentage"]
    ]
    v = v.merge(m, on="video_id", how="left")

    total = t.daily.groupby("video_id")["engagedViews"].sum()
    post = t.daily[t.daily["day"] >= C.VIEW_CHANGE_DATE].groupby("video_id")["engagedViews"].sum()
    share = post.reindex(total.index, fill_value=0) / total.where(total > 0)
    v["post_change_share"] = v["video_id"].map(share)

    r = t.retention.sort_values(["video_id", "elapsedVideoTimeRatio"])
    curves = r.groupby("video_id").agg(
        curve_points=("audienceWatchRatio", "count"),
        mean_rrp=("relativeRetentionPerformance", "mean"),
    )
    v = v.merge(curves, left_on="video_id", right_index=True, how="left")
    v["curve_points"] = v["curve_points"].fillna(0).astype(int)

    xy = {vid: (g["elapsedVideoTimeRatio"].to_numpy(), g["audienceWatchRatio"].to_numpy())
          for vid, g in r.groupby("video_id")}

    def hook3s(row) -> float:
        if row.video_id not in xy or not row.duration_seconds:
            return np.nan
        x, y = xy[row.video_id]
        return float(np.interp(min(1.0, C.HOOK_SECONDS / row.duration_seconds), x, y))

    v["hook3s"] = v.apply(hook3s, axis=1)
    return v.reset_index(drop=True)


def eligible_pool(
    frame: pd.DataFrame, available: set[str] | None, max_post_share: float
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply rules 2.1 in order; return the pool and per-rule exclusion counts."""
    rules = [
        ("category_" + C.REFERENCE_CATEGORY, frame["category"] == C.REFERENCE_CATEGORY),
        ("duration_8_60s", frame["duration_seconds"].between(C.MIN_DURATION_S, C.MAX_DURATION_S)),
        ("published_before_" + C.PUBLISHED_BEFORE, _ts(frame["published_at"]) < pd.Timestamp(C.PUBLISHED_BEFORE, tz="UTC")),
        (f"post_change_share_le_{max_post_share:.2f}", frame["post_change_share"] <= max_post_share),
        (f"engaged_views_ge_{C.MIN_ENGAGED_VIEWS}", frame["engagedViews"] >= C.MIN_ENGAGED_VIEWS),
        ("complete_retention_curve", frame["curve_points"] == C.CURVE_POINTS),
    ]
    if available is not None:
        rules.append(("video_file_available", frame["video_id"].isin(available)))
    keep = pd.Series(True, index=frame.index)
    funnel = [{"rule": "all_shorts", "excluded": 0, "remaining": len(frame)}]
    for name, mask in rules:
        new = keep & mask.fillna(False).astype(bool)
        funnel.append({"rule": name, "excluded": int((keep & ~new).sum()), "remaining": int(new.sum())})
        keep = new
    if available is None:
        funnel.append({"rule": "video_file_available", "excluded": 0, "remaining": int(keep.sum()), "note": "not applied (no --available)"})
    return frame[keep].copy(), funnel


def label_candidates(pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """hold_resid, top/bottom quartile labels, and the 3x3 duration x date grid."""
    pool = pool.copy()
    pool["hold_score"] = pool["averageViewPercentage"]
    pool["hold_resid"] = hold_resid(pool, pd.Series(True, index=pool.index))
    lo, hi = pool["hold_resid"].quantile([C.CANDIDATE_QUANTILE, 1 - C.CANDIDATE_QUANTILE])
    pool["label"] = None
    pool.loc[pool["hold_resid"] >= hi, "label"] = "strong"
    pool.loc[pool["hold_resid"] <= lo, "label"] = "weak"
    pool["dur_tertile"] = pd.qcut(pool["duration_seconds"].rank(method="first"), 3, labels=False)
    pool["date_tertile"] = pd.qcut(_ts(pool["published_at"]).rank(method="first"), 3, labels=False)
    info = {
        "hold_resid_cutoffs": {"weak_le": float(lo), "strong_ge": float(hi)},
        "candidates": {"strong": int((pool["label"] == "strong").sum()), "weak": int((pool["label"] == "weak").sum())},
    }
    return pool, info
