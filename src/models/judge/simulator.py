"""The retention simulator: one predicted audience curve per draft (plan/03d).

Three segment ratios (early 2-25 %, mid 25-75 %, late 75-95 % of the runtime) come from three specialists, each
calibrated as  ratio = reference ratio + a + b * signal.  The shape inside a segment comes from the per-second
model. With no specialist and no shape the output is exactly the reference curve (what a Short of that length
does on this channel).

    ./venv/bin/python -m src.models.judge.simulator onecall    # S2: can one v3 extraction feed the models?
    ./venv/bin/python -m src.models.judge.simulator build      # S3 + S4 on the 145, everything out of fold
    ./venv/bin/python -m src.models.judge.simulator gate       # S6: check of the frozen configuration on GATE
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.judge.library import ANCHOR_N, JUDGE_DIR, K, SEED, Library, build, oof_folds, rf_late_oof

GRID = np.arange(1, 101) / 100                       # r = 0.01 ... 1.00
I2, I25, I75, I95 = 1, 24, 74, 94                    # indices of 2 %, 25 %, 75 %, 95 %
SEGMENTS = {"early": (I2, I25), "mid": (I25, I75), "late": (I75, I95)}
RATIO_CLIP = (0.2, 1.6)


def normalise(curves: np.ndarray) -> np.ndarray:
    """Viewers per 100 present at 2 % of the runtime."""
    return 100 * curves / curves[:, [I2]]


def raw_curves(retention: pd.DataFrame, ids) -> np.ndarray:
    g = retention[retention["video_id"].isin(set(ids))].sort_values(["video_id", "elapsedVideoTimeRatio"])
    by = {v: x["audienceWatchRatio"].to_numpy() for v, x in g.groupby("video_id")}
    return np.vstack([by[v] for v in ids])


def reference(curves: np.ndarray, durations: np.ndarray, pool: np.ndarray, d: float, exclude=()) -> np.ndarray:
    """Mean normalised curve of the ANCHOR_N pool videos closest in length (never the excluded ones)."""
    p = np.setdiff1d(pool, list(exclude))
    return curves[p[np.argsort(np.abs(durations[p] - d), kind="stable")][:ANCHOR_N]].mean(axis=0)


def ratios(curve: np.ndarray) -> np.ndarray:
    """(..., 3): early, mid, late ratios of a curve or a stack of curves."""
    return np.stack([curve[..., b] / curve[..., a] for a, b in SEGMENTS.values()], axis=-1)


def assemble(ref: np.ndarray, targets: dict[str, float] | None = None, shape_loss: np.ndarray | None = None) -> np.ndarray:
    """100-point curve. ``targets``: segment -> ratio (missing segments keep the reference's ratio).
    ``shape_loss``: per-step relative losses (99) from the per-second model; None keeps the reference's shape."""
    targets = targets or {}
    g_ref = np.log(ref[:-1]) - np.log(ref[1:])
    g = g_ref.copy() if shape_loss is None else -np.log(1 - np.clip(shape_loss, -0.5, 0.9))
    for name, (a, b) in SEGMENTS.items():
        t = float(np.clip(targets.get(name, ref[b] / ref[a]), *RATIO_CLIP))
        g[a:b] += (-np.log(t) - g[a:b].sum()) / (b - a)
    if shape_loss is not None:                       # outside the three segments the shape is not anchored: keep the reference
        g[:I2], g[I95:] = g_ref[:I2], g_ref[I95:]
    logw = np.empty(100)
    logw[I2] = np.log(100.0)
    logw[I2 + 1:] = logw[I2] - np.cumsum(g[I2:])
    logw[0] = logw[I2] + g[0]
    return np.exp(logw)


def fit_calibration(signal: np.ndarray, deviation: np.ndarray) -> tuple[float, float]:
    """ratio deviation = a + b * signal by least squares; a negative slope is not trusted and becomes 0."""
    X = np.column_stack([np.ones(len(signal)), signal])
    a, b = np.linalg.lstsq(X, deviation, rcond=None)[0]
    return (float(deviation.mean()), 0.0) if b < 0 else (float(a), float(b))
