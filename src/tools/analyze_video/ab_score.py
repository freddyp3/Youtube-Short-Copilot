"""Score the v2-vs-v3 consistency A/B (plan/03b section 4). Pre-registered: committed before any A/B call.

Reads only the raw extraction records, the selection csv and the cut files; never loads an outcome column.

    ./venv/bin/python -m src.tools.analyze_video.ab_score --v2 data/features/gemini/raw/<run_id_v2> \
        --v3 data/features/gemini/raw/<run_id_v3> [--v3-read-ok yes|no]
"""

from __future__ import annotations

import numpy as np

SEED, N_BOOT = 340, 2000
ICC_MIN, KAPPA_MIN, TS_MAX_MEDIAN_S, TS_WINDOW_S = 0.7, 0.6, 1.0, 1.0

# core features (03b 4.2); "tail" and the main-payoff timestamp resolve to a variant (raw + tail_all for step 2)
CORE = ["t_premise_clear_s", "payoff_count", "t_main_payoff_s", "tail", "dead_time_s", "arc_ord", "loop_seam"]


# ----------------------------------------------------------------- statistics

def icc21(*cols) -> float:
    """ICC(2,1), two-way random effects, absolute agreement, single rater (Shrout & Fleiss 1979); one column per run."""
    d = np.column_stack(cols).astype(float)
    n, k = d.shape
    if n < 2:
        return float("nan")
    g = d.mean()
    ssr = k * ((d.mean(axis=1) - g) ** 2).sum()
    ssc = n * ((d.mean(axis=0) - g) ** 2).sum()
    sst = ((d - g) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    if den <= 0:
        return 1.0 if sst == 0 else float("nan")
    return float((msr - mse) / den)


def kappa(a, b, levels=None, weighted: bool = False) -> float:
    """Cohen's kappa; linear weights when ``weighted`` (levels give the order). Both runs constant and equal -> 1."""
    a, b = list(a), list(b)
    levels = list(levels) if levels is not None else sorted({str(v) for v in a + b})
    idx = {str(lv): i for i, lv in enumerate(levels)}
    L = len(levels)
    m = np.zeros((L, L))
    for u, v in zip(a, b):
        m[idx[str(u)], idx[str(v)]] += 1
    m /= m.sum()
    if weighted and L > 1:
        i, j = np.indices((L, L))
        w = 1 - np.abs(i - j) / (L - 1)
    else:
        w = np.eye(L)
    po = (w * m).sum()
    pe = (w * np.outer(m.sum(axis=1), m.sum(axis=0))).sum()
    if np.isclose(1 - pe, 0):
        return 1.0 if np.isclose(po, 1) else float("nan")
    return float((po - pe) / (1 - pe))


def ts_stats(x, y) -> tuple[float, float]:
    """(median |dt|, share of |dt| <= TS_WINDOW_S)."""
    d = np.abs(np.asarray(x, float) - np.asarray(y, float))
    return float(np.median(d)), float((d <= TS_WINDOW_S).mean())


def stat(kind: str, x, y, levels=None) -> dict:
    if kind == "icc":
        return {"icc": icc21(x, y)}
    if kind == "kappa":
        return {"kappa": kappa(x, y)}
    if kind == "wkappa":
        return {"kappa": kappa(x, y, levels, weighted=True)}
    med, share = ts_stats(x, y)
    return {"median_abs_dt": med, "share_within_1s": share}


def headline(kind: str, s: dict) -> float:
    """The value compared with the threshold and used in the tie-break mean (timestamps: share within 1 s)."""
    return s["icc"] if kind == "icc" else s["kappa"] if kind in ("kappa", "wkappa") else s["share_within_1s"]


def passes(kind: str, s: dict) -> bool:
    if kind == "icc":
        return s["icc"] >= ICC_MIN
    if kind in ("kappa", "wkappa"):
        return s["kappa"] >= KAPPA_MIN
    return s["median_abs_dt"] <= TS_MAX_MEDIAN_S


def bootstrap_ci(kind: str, x, y, levels=None) -> list[float]:
    x, y = np.asarray(x, dtype=object), np.asarray(y, dtype=object)
    rng = np.random.default_rng(SEED)
    vals = []
    for idx in rng.integers(0, len(x), (N_BOOT, len(x))):
        vals.append(headline(kind, stat(kind, x[idx], y[idx], levels)))
    vals = np.asarray(vals, float)
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]
