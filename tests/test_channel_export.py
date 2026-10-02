"""Tests for src.channel_export using mocked googleapiclient service objects (no network)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httplib2
import pandas as pd
import pytest
from googleapiclient.errors import HttpError

from src.channel_export import analytics, auth, data_api, reporting
from src.channel_export.common import ApiCaller, CallStats, RawStore, execute_with_retry
from src.channel_export.flatten import (
    comments_to_df,
    content_types_by_video,
    parse_iso_duration,
    result_table_to_df,
    videos_to_df,
    write_csv,
)

# ---------------------------------------------------------------- fakes


class FakeRequest:
    """Stands in for a googleapiclient HttpRequest."""

    def __init__(self, response: Any = None, errors: list[Exception] | None = None) -> None:
        self.response = response
        self.errors = list(errors or [])
        self.calls = 0

    def execute(self, num_retries: int = 0) -> Any:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.response


def http_error(
    status: int, reason: str = "badRequest", message: str = "The query is not supported."
) -> HttpError:
    resp = httplib2.Response({"status": status})
    resp.reason = message
    body = {"error": {"code": status, "message": message, "errors": [{"reason": reason, "message": message}]}}
    return HttpError(resp, json.dumps(body).encode())


def no_sleep_caller(tmp_path: Path | None = None) -> ApiCaller:
    return ApiCaller(RawStore(tmp_path / "raw" if tmp_path else None), sleep=lambda _s: None)


# ---------------------------------------------------------------- retries / quota


def test_execute_with_retry_backs_off_on_5xx_and_429() -> None:
    request = FakeRequest(
        {"ok": True}, errors=[http_error(503, "backendError"), http_error(429, "rateLimitExceeded")]
    )
    sleeps: list[float] = []
    assert execute_with_retry(request, sleep=sleeps.append, base_delay=1.0) == {"ok": True}
    assert request.calls == 3
    assert len(sleeps) == 2 and 1.0 <= sleeps[0] < 2.0 and 2.0 <= sleeps[1] < 3.0


def test_execute_with_retry_does_not_retry_400_or_quota() -> None:
    for err in (http_error(400), http_error(403, "quotaExceeded")):
        request = FakeRequest(errors=[err])
        with pytest.raises(HttpError):
            execute_with_retry(request, sleep=lambda _s: None)
        assert request.calls == 1


def test_call_stats_counts_retries_as_quota_units(tmp_path: Path) -> None:
    caller = no_sleep_caller(tmp_path)
    caller.execute(
        "youtube",
        "videos.list",
        FakeRequest({"items": []}, errors=[http_error(500, "backendError")]),
        "x.json",
    )
    caller.execute("analytics", "reports.query", FakeRequest({}), None)
    summary = caller.stats.summary()
    assert summary["data_api_units_used"] == 2
    assert summary["analytics_api_calls"] == 1
    assert (tmp_path / "raw" / "x.json").exists()


# ---------------------------------------------------------------- Data API


def test_list_uploads_follows_page_tokens(tmp_path: Path) -> None:
    pages = {
        None: {"items": [{"contentDetails": {"videoId": f"v{i}"}} for i in range(50)], "nextPageToken": "p2"},
        "p2": {
            "items": [{"contentDetails": {"videoId": f"v{i}"}} for i in range(50, 100)],
            "nextPageToken": "p3",
        },
        "p3": {"items": [{"contentDetails": {"videoId": "v100"}}]},
    }
    youtube = MagicMock()
    youtube.playlistItems.return_value.list.side_effect = lambda **kw: FakeRequest(pages[kw.get("pageToken")])
    items = data_api.list_uploads(youtube, no_sleep_caller(tmp_path), "UUabc")
    ids = data_api.upload_video_ids(items)
    assert ids == [f"v{i}" for i in range(101)]
    calls = youtube.playlistItems.return_value.list.call_args_list
    assert [c.kwargs.get("pageToken") for c in calls] == [None, "p2", "p3"]
    assert all(c.kwargs["maxResults"] == 50 and c.kwargs["playlistId"] == "UUabc" for c in calls)
    assert len(list((tmp_path / "raw" / "data_api" / "playlistItems.list").glob("page_*.json"))) == 3


def test_list_uploads_stops_at_max_videos() -> None:
    pages = {
        None: {"items": [{"contentDetails": {"videoId": f"v{i}"}} for i in range(50)], "nextPageToken": "p2"}
    }
    youtube = MagicMock()
    youtube.playlistItems.return_value.list.side_effect = lambda **kw: FakeRequest(pages[kw.get("pageToken")])
    items = data_api.list_uploads(youtube, no_sleep_caller(), "UUabc", max_videos=7)
    assert len(items) == 7
    assert youtube.playlistItems.return_value.list.call_count == 1


def test_get_videos_batches_ids_in_fifties() -> None:
    youtube = MagicMock()

    def videos_list(**kw: Any) -> FakeRequest:
        return FakeRequest({"items": [{"id": vid} for vid in kw["id"].split(",")]})

    youtube.videos.return_value.list.side_effect = videos_list
    ids = [f"v{i}" for i in range(120)]
    videos, dropped = data_api.get_videos(youtube, no_sleep_caller(), ids)
    calls = youtube.videos.return_value.list.call_args_list
    assert [len(c.kwargs["id"].split(",")) for c in calls] == [50, 50, 20]
    assert all(c.kwargs["part"] == ",".join(data_api.VIDEO_PARTS) for c in calls)
    assert [v["id"] for v in videos] == ids
    assert dropped == []


def test_get_videos_drops_parts_that_error() -> None:
    youtube = MagicMock()

    def videos_list(**kw: Any) -> FakeRequest:
        if "brandPartner" in kw["part"].split(","):
            return FakeRequest(errors=[http_error(400, "invalidPart")])
        return FakeRequest({"items": [{"id": vid} for vid in kw["id"].split(",")]})

    youtube.videos.return_value.list.side_effect = videos_list
    videos, dropped = data_api.get_videos(youtube, no_sleep_caller(), ["a", "b"])
    assert dropped == ["brandPartner"]
    assert [v["id"] for v in videos] == ["a", "b"]
    assert "brandPartner" not in youtube.videos.return_value.list.call_args_list[-1].kwargs["part"]


def _comment(cid: str, text: str, parent: str | None = None) -> dict[str, Any]:
    snippet: dict[str, Any] = {"textDisplay": text, "authorDisplayName": "x", "likeCount": 1}
    if parent:
        snippet["parentId"] = parent
    return {"id": cid, "snippet": snippet}


def test_comments_paginate_and_fetch_missing_replies() -> None:
    thread_pages = {
        None: {
            "items": [
                {
                    "id": "t1",
                    "snippet": {"topLevelComment": _comment("t1", "first"), "totalReplyCount": 3},
                    "replies": {"comments": [_comment("t1.r1", "r1", "t1")]},
                }
            ],
            "nextPageToken": "n2",
        },
        "n2": {
            "items": [
                {"id": "t2", "snippet": {"topLevelComment": _comment("t2", "second"), "totalReplyCount": 0}}
            ]
        },
    }
    reply_pages = {
        None: {
            "items": [_comment("t1.r1", "r1", "t1"), _comment("t1.r2", "r2", "t1")],
            "nextPageToken": "r2",
        },
        "r2": {"items": [_comment("t1.r3", "r3", "t1")]},
    }
    youtube = MagicMock()

    def threads_list(**kw: Any) -> FakeRequest:
        if kw["videoId"] == "disabled":
            return FakeRequest(errors=[http_error(403, "commentsDisabled", "Comments disabled")])
        return FakeRequest(thread_pages[kw.get("pageToken")])

    youtube.commentThreads.return_value.list.side_effect = threads_list
    youtube.comments.return_value.list.side_effect = lambda **kw: FakeRequest(
        reply_pages[kw.get("pageToken")]
    )

    errors: list[dict[str, Any]] = []
    records = data_api.fetch_comments(youtube, no_sleep_caller(), ["vid1", "disabled"], errors)

    assert errors == []  # commentsDisabled is not an error
    thread_calls = youtube.commentThreads.return_value.list.call_args_list
    assert [c.kwargs.get("pageToken") for c in thread_calls if c.kwargs["videoId"] == "vid1"] == [None, "n2"]
    assert all(c.kwargs["maxResults"] == 100 and c.kwargs["part"] == "snippet,replies" for c in thread_calls)
    reply_calls = youtube.comments.return_value.list.call_args_list
    assert [c.kwargs["parentId"] for c in reply_calls] == ["t1", "t1"]  # only t1 needed the full fetch

    df = comments_to_df(records)
    assert len(df) == 5  # 2 top-level + 3 replies
    assert df["is_reply"].sum() == 3
    assert set(df["video_id"]) == {"vid1"}
    assert df.loc[df["comment_id"] == "t1.r3", "thread_id"].item() == "t1"


# ---------------------------------------------------------------- Analytics flattening + pagination


def test_result_table_to_df_and_csv_round_trip(tmp_path: Path) -> None:
    response = {
        "kind": "youtubeAnalytics#resultTable",
        "columnHeaders": [
            {"name": "day", "columnType": "DIMENSION", "dataType": "STRING"},
            {"name": "views", "columnType": "METRIC", "dataType": "INTEGER"},
            {"name": "averageViewPercentage", "columnType": "METRIC", "dataType": "FLOAT"},
        ],
        "rows": [["2026-09-01", 120, 88.5], ["2026-09-02", 7, 101.25]],
    }
    df = result_table_to_df(response)
    assert list(df.columns) == ["day", "views", "averageViewPercentage"]
    assert str(df["views"].dtype) == "Int64" and df["views"].sum() == 127
    assert df["averageViewPercentage"].dtype == "float64"
    assert write_csv(df, tmp_path / "out.csv") == 2
    back = pd.read_csv(tmp_path / "out.csv")
    assert back.to_dict("records")[1] == {"day": "2026-09-02", "views": 7, "averageViewPercentage": 101.25}


def test_result_table_without_rows_keeps_columns() -> None:
    df = result_table_to_df(
        {
            "columnHeaders": [
                {"name": "country", "dataType": "STRING"},
                {"name": "views", "dataType": "INTEGER"},
            ]
        }
    )
    assert df.empty and list(df.columns) == ["country", "views"]


def test_run_query_paginates_top_videos_with_start_index() -> None:
    headers = [{"name": "video", "dataType": "STRING"}, {"name": "views", "dataType": "INTEGER"}]
    rows = [[f"v{i}", 1000 - i] for i in range(250)]
    service = MagicMock()

    def query(**params: Any) -> FakeRequest:
        start = params["startIndex"] - 1
        return FakeRequest({"columnHeaders": headers, "rows": rows[start : start + params["maxResults"]]})

    service.reports.return_value.query.side_effect = query
    spec = analytics.ReportSpec(
        name="top", metrics=("views",), dimensions=("video",), sort="-views", max_results=200, paginate=True
    )
    merged = analytics.run_query(
        service, no_sleep_caller(), spec, date(2025, 1, 1), date(2026, 1, 1), "analytics/top"
    )
    assert len(merged["rows"]) == 250
    calls = service.reports.return_value.query.call_args_list
    assert [c.kwargs["startIndex"] for c in calls] == [1, 201]
    assert all(c.kwargs["sort"] == "-views" and c.kwargs["ids"] == "channel==MINE" for c in calls)


def test_plan_traffic_batches_respects_video_day_limit() -> None:
    end = date(2026, 9, 25)
    videos = [(f"v{i}", date(2026, 1, 1)) for i in range(10)] + [("old", date(2020, 1, 1))]
    batches = analytics.plan_traffic_batches(videos, end, limit=1000)
    for ids, start in batches:
        assert len(ids) * ((end - start).days + 1) <= 1000 or len(ids) == 1
    assert sorted(v for ids, _ in batches for v in ids) == sorted(v for v, _ in videos)


def test_videos_to_df_labels_shorts() -> None:
    metrics = pd.DataFrame(
        {
            "video_id": ["a", "b", "b"],
            "creatorContentType": ["SHORTS", "VIDEO_ON_DEMAND", "LIVE_STREAM"],
            "views": [5, 9, 1],
        }
    )
    types = content_types_by_video(metrics)
    assert types == {"a": "SHORTS", "b": "VIDEO_ON_DEMAND"}
    videos = [
        {"id": "a", "snippet": {"title": "A", "tags": ["x", "y"]}, "contentDetails": {"duration": "PT45S"}},
        {"id": "b", "snippet": {"title": "B"}, "contentDetails": {"duration": "PT1H2M3S"}},
        {"id": "c", "snippet": {"title": "C"}, "contentDetails": {"duration": "P0D"}},
    ]
    df = videos_to_df(videos, types)
    assert df["is_short"].tolist()[:2] == [True, False] and pd.isna(df["is_short"].iloc[2])
    assert df["duration_seconds"].tolist() == [45.0, 3723.0, 0.0]
    assert json.loads(df.loc[0, "snippet.tags"]) == ["x", "y"]
    assert parse_iso_duration("bogus") is None
