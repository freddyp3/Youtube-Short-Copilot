"""Gemini client for the hook test: upload cache, generate wrapper, retry, usage+cost.

Free tier target ($0): ``tier`` comes from config/``.env`` (``GEMINI_TIER``);
``cost_usd_actual`` is 0 on free while ``cost_usd_paid_equiv`` records what the
call would cost at paid prices. Rate limits: 429s honour any retry delay found
in the error, else exponential backoff (cap ~120 s); a per-day-quota error or
429s persisting past MAX_RETRIES raises ``DailyQuotaExceeded`` so the run stops
cleanly and can be resumed with the same ``--run-id``.

SDK notes (google-genai 2.32.0, verified in types.py):
- per-video fps: ``Part(file_data=..., video_metadata=types.VideoMetadata(fps=5.0))``
- per-part media resolution: ``PartMediaResolution(level=PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH)``
- Files API: ``client.files.upload(file=path, config={"display_name": ...})``,
  poll ``client.files.get(name=...)`` until ``file.state == "ACTIVE"``; ``file.uri``
  feeds ``FileData.file_uri``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.evals.hook_test import config as C

DAILY_QUOTA_RE = re.compile(r"per.?day|daily|PerDay|RPD|RequestsPerDay", re.I)
RETRY_DELAY_RE = re.compile(r"retryDelay\"?\s*[:=]\s*\"?(\d+(?:\.\d+)?)s", re.I)


class DailyQuotaExceeded(Exception):
    """Daily quota hit or 429s past MAX_RETRIES; rerun with --run-id to resume."""


@dataclass
class CallResult:
    record: dict = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def nominal_cost(model: str, prompt_tokens: int, output_tokens: int) -> float:
    """Paid-equivalent USD for one call."""
    pin, pout = C.PRICES_USD_PER_M.get(model, (0.0, 0.0))
    return prompt_tokens * pin / 1e6 + output_tokens * pout / 1e6


def estimate_call_cost(model: str, duration_s: float, fps: float = C.FPS) -> float:
    """Nominal cost estimate when count_tokens is unavailable (plan 4.1 token model)."""
    tin = duration_s * fps * C.ASSUMED_TOKENS_PER_FRAME + duration_s * C.AUDIO_TOKENS_PER_S \
        + C.PROMPT_TOKENS_EST
    return nominal_cost(model, int(tin), C.OUTPUT_TOKENS_EST)


def is_rate_limit(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    return code == 429 or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)


def is_daily_quota(e: Exception) -> bool:
    blob = str(e) + json.dumps(getattr(e, "details", {}) or {}, default=str)
    return DAILY_QUOTA_RE.search(blob) is not None


def retry_delay_s(e: Exception, attempt: int) -> float:
    blob = str(e) + json.dumps(getattr(e, "details", {}) or {}, default=str)
    m = RETRY_DELAY_RE.search(blob)
    if m:
        return min(float(m.group(1)), 120.0)
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", blob, re.I)
    if m:
        return min(float(m.group(1)), 120.0)
    return min(5.0 * 2**attempt, 120.0)


def extract_text(resp) -> str:
    """JSON response text, excluding thought-summary parts.

    In 2.32.0 ``resp.text`` already skips parts with ``part.thought == True``
    (GenerateContentResponse._get_text); we filter again in the fallback path
    so a mocked/edge response can't leak the scratch-pad summary into the JSON.
    """
    t = getattr(resp, "text", None)
    if t:
        return t
    try:
        return "".join(
            getattr(p, "text", "") or ""
            for c in resp.candidates for p in c.content.parts
            if getattr(p, "thought", None) is not True)
    except Exception:
        return ""


def extract_thought_summary(resp) -> str:
    """Joined thought-summary parts (include_thoughts=True)."""
    try:
        return "\n".join(
            getattr(p, "text", "") or ""
            for c in resp.candidates for p in c.content.parts
            if getattr(p, "thought", None) is True)
    except Exception:
        return ""


def extract_usage(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return {}
    g = lambda k: getattr(u, k, None) or 0
    return {
        "prompt_tokens": g("prompt_token_count"),
        "response_tokens": g("candidates_token_count") or g("response_token_count"),
        "thinking_tokens": g("thoughts_token_count"),
        "total_tokens": g("total_token_count"),
    }


def new_genai_client(api_key: str):
    """Import-time-free construction so tests never need the SDK or a key."""
    from google import genai
    from google.genai import types
    # SDK-level retries off: _call_with_retry owns retry timing (each retry spends free-tier quota)
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=1)))
