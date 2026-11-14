"""predict_retention: how the channel's audience is expected to watch a draft Short (plan/03d sections 3.3, 12).

    ./venv/bin/python -m src.tools.predict_retention.tool path/to/draft.mp4 [--out result.json]

Five specialists, each used only for the item it was confirmed on (GATE and HELDOUT):
  drop moments      Gemini reading the draft's one-second timeline
  mid-video hold    mean of the two Gemini calls (draft only; with similar past Shorts)
  end-of-video hold random forest on the structure features
  early hold        random forest on the structure features
  rewatch level     Gemini shown similar past Shorts (similarity >= 0.75) with their real outcomes
The library is every labelled Short of the channel (235). The curve is a drawing of those statements on the
typical curve for the draft's length; it is not a number-for-number forecast.
Costs four Gemini calls per draft (two extraction, two judging) on GEMINI_API_KEY_PAID.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evals.hook_test import config as C
from src.evals.hook_test.data import load_export
from src.evals.hook_test.gemini_client import new_genai_client, sha256_file
from src.models import cv, hazard as H, targets as T
from src.models.dataset import load_split
from src.models.experiments import all_recipes
from src.models.features import M1_COLS, design, prepare
from src.models.judge import score as S, simulator as M
from src.models.judge.gate import RUNS
from src.models.judge.library import (JUDGE_DIR, SEED, TIERS, V3_RUNS, Library, build_from, embed_timelines,
                                      load_timelines, oof_folds, rf_late_oof, timeline_lines, chunked_embeddings, ENCODER)
from src.tools.analyze_video.ab_score import ARC_ORD
from src.tools.analyze_video.derive import derive_features

SPLITS = ("train", "dev", "gate", "heldout")
SETS = ("pilot", "validation", "gate", "heldout")
SEGMENT_TARGET = {"early": "early_hold", "late": "late_hold"}


@dataclass
class State:
    lib: Library
    emb: np.ndarray
    export: object
    curves: np.ndarray              # (n, 100) real normalised curves
    ref: np.ndarray                 # (n, 100) leave-one-out reference curves
    signal: dict[str, np.ndarray]   # segment -> library signals made without each video's own outcome
    calib: dict[str, tuple[float, float]]
    band: np.ndarray                # (2, 5) residual percentiles at the checkpoints
    ps_loss: np.ndarray
    seg: object


_STATE: State | None = None


def library_frame() -> tuple[pd.DataFrame, object]:
    df = pd.concat([load_split(s) for s in SPLITS], ignore_index=True)
    export = load_export(Path(json.loads(C.SELECTION_JSON_PATH.read_text())["export_dir"]))
    return T.add_targets(df, export), export


def gemini_points(ids: list[str], tag: str) -> np.ndarray:
    g = pd.concat([S.load_run(f"{tag}_{s}") for s in SETS]).set_index("video_id").loc[ids]
    return g[["p10", "p25", "p50", "p75", "p95"]].to_numpy(dtype=float)


def mid_signal(p_a: np.ndarray, p_b: np.ndarray, ref: np.ndarray) -> np.ndarray:
    m = (p_a + p_b) / 2
    return m[:, 3] / m[:, 1] - ref[:, M.I75] / ref[:, M.I25]


def state() -> State:
    """Library, models' out-of-fold signals, calibration and band for the 235 videos. Built once per process."""
    global _STATE
    if _STATE is not None:
        return _STATE
    df, export = library_frame()
    timelines = load_timelines(df, runs=V3_RUNS + tuple(RUNS.values()))
    lib = build_from(df, export, JUDGE_DIR / "timeline_embeddings_all.npz", timelines)
    n, ids = len(lib), lib.df["video_id"].tolist()
    emb = embed_timelines(lib.df, timelines, JUDGE_DIR / "timeline_embeddings_all.npz")
    curves = M.normalise(M.raw_curves(export.retention, ids))
    dur = lib.df["duration_s"].to_numpy(dtype=float)
    ref = np.vstack([M.reference(curves, dur, np.arange(n), dur[i], exclude=[i]) for i in range(n)])
    data, folds = prepare(lib.df), oof_folds(n)
    recipe = all_recipes(data)["M1_core_mechanics"]
    signal = {s: rf_late_oof(data, lib.df[t].to_numpy(dtype=float), recipe, folds) for s, t in SEGMENT_TARGET.items()}
    signal["mid"] = mid_signal(gemini_points(ids, "j1_L0_E_3.8-flash"), gemini_points(ids, "j1_L2_T75_3.8-flash"), ref)
    seg = H.build(lib.df, export.retention, lib.payoffs)
    ps = H.oof(seg, np.column_stack([seg.base, seg.cov]), "ridge", folds).reshape(n, -1)
    pieces = M.Pieces(ids, dur, curves, ref, signal, ps, lib.df)
    pred = M.oof_predict(pieces, tuple(M.SEGMENTS), True)
    _STATE = State(lib, emb, export, curves, ref, signal, M.calibrate(pieces, np.arange(n)), M.band_coverage(pred, pieces)[1], ps, seg)
    return _STATE


def blind_copy(path: Path, out_dir: Path) -> Path:
    """Copy without container metadata, so no title or date travels with the file."""
    out = out_dir / f"draft_{sha256_file(path)[:10]}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-map_metadata", "-1", "-c", "copy", str(out)], check=True)
    return out


def extract(path: Path, blind_id: str, dur: float, version: str) -> dict:
    from src.tools.analyze_video import extract as E
    from src.tools.analyze_video.schema import render_prompt

    pf = E.PROMPTS_DIR / f"{version}.md"
    rec = E.GeminiFeatureExtractor(model=C.DEFAULT_MODEL, version=version).analyze(
        blind_id, path, render_prompt(pf.read_text(), dur), dur, fps=C.FPS, media_resolution=C.MEDIA_RESOLUTION,
        prompt_version=version, prompt_sha256=sha256_file(pf))
    if rec["status"] != "ok":
        raise RuntimeError(f"extraction {version} failed: {rec.get('invalid_reason')}")
    return rec


def draft_row(v2: dict, code: dict, dur: float, template: pd.Series) -> pd.Series:
    """One library-shaped row for the draft: structure features from the v2 extraction and the code measurements."""
    d = derive_features(v2["result"], dur, [])
    row = template.copy()
    row[:] = np.nan
    for k, v in {**d, **code}.items():
        if k in row.index:
            row[k] = v
    row["video_id"], row["blind_id"], row["split"] = "draft", "draft", "draft"
    row["duration_s"], row["duration_seconds"] = dur, float(np.ceil(dur))
    row["published_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row["t_premise_clear_s"] = d["t_premise_clear_s"] if d["t_premise_clear_s"] is not None else dur
    row["arc_ord"] = ARC_ORD[d["arc"]]
    row["is_short"] = float(dur <= 20.0)
    return row


def with_draft(st: State, row: pd.Series, timeline: list[dict], payoffs: list[dict]) -> Library:
    """Library plus the draft as the last row. The draft is outside the pool and has no outcomes."""
    from sentence_transformers import SentenceTransformer

    lib, n = st.lib, len(st.lib)
    df = pd.concat([lib.df, row.to_frame().T], ignore_index=True)
    for c in lib.df.columns:
        if pd.api.types.is_numeric_dtype(lib.df[c]):
            df[c] = pd.to_numeric(df[c])
    model = SentenceTransformer(ENCODER)
    e = chunked_embeddings([timeline_lines(timeline)], lambda p: model.encode(p, normalize_embeddings=True))
    emb = np.vstack([st.emb, e])
    X = prepare(df).matrix(M1_COLS)
    Z = (X - X[:n].mean(axis=0)) / X[:n].std(axis=0)
    pad = lambda a, v: np.concatenate([a, np.full((1,) + a.shape[1:], v)])
    return Library(df=df, timelines={**lib.timelines, "draft": timeline}, payoffs={**lib.payoffs, "draft": payoffs},
                   points=pad(lib.points, np.nan), rewatch=np.append(lib.rewatch, "nan"), likes=np.append(lib.likes, "nan"),
                   late=np.append(lib.late, np.nan), sim_e=emb @ emb.T,
                   dist_s=np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)),
                   hint_points=pad(lib.hint_points, np.nan), hint_late_pct=np.append(lib.hint_late_pct, np.nan),
                   split={"draft": ["draft"]}, pool=np.arange(n))
