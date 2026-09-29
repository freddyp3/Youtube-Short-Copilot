"""YouTube Analytics API v2 (reports.query) report definitions for an owned channel.

Every dimension/metric/filter combination below was checked on 2026-09-25 against
https://developers.google.com/youtube/analytics/channel_reports (updated 2026-09-14),
/metrics, /dimensions and /revision_history. Notes:

* ``views`` for Shorts counts every start/replay (since 2025-04-30); ``engagedViews`` keeps
  the old definition. Both are requested everywhere they are allowed.
* ``creatorContentType`` is documented only as an optional *dimension* (never as a filter), so
  Shorts splits are produced by adding it as a dimension instead of ``creatorContentType==SHORTS``.
* Card/annotation metrics are omitted: cards and annotations don't exist on Shorts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

from .common import ApiCaller

log = logging.getLogger(__name__)

IDS = "channel==MINE"

VIEW_METRICS: tuple[str, ...] = (
    "engagedViews",
    "views",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
)
PREMIUM_METRICS: tuple[str, ...] = ("redViews", "estimatedRedMinutesWatched")
ENGAGEMENT_METRICS: tuple[str, ...] = (
    "likes",
    "dislikes",
    "comments",
    "shares",
    "subscribersGained",
    "subscribersLost",
    "videosAddedToPlaylists",
    "videosRemovedFromPlaylists",
)
# Valid for: Top videos, Time-based, User activity by country (channel_reports).
BASIC_METRICS: tuple[str, ...] = VIEW_METRICS + PREMIUM_METRICS + ENGAGEMENT_METRICS
# "User activity by subscribed status" does not list comments/subscribersGained/Lost.
SUBSCRIBED_STATUS_METRICS: tuple[str, ...] = VIEW_METRICS + (
    "likes",
    "dislikes",
    "shares",
    "videosAddedToPlaylists",
    "videosRemovedFromPlaylists",
)
# Traffic source, traffic source detail, device/OS and playback location reports.
INSIGHT_METRICS: tuple[str, ...] = ("engagedViews", "views", "estimatedMinutesWatched")
# Need the yt-analytics-monetary.readonly scope and a YPP channel (403 otherwise).
MONETARY_METRICS: tuple[str, ...] = (
    "estimatedRevenue",
    "estimatedAdRevenue",
    "grossRevenue",
    "estimatedRedPartnerRevenue",
    "monetizedPlaybacks",
    "playbackBasedCpm",
    "adImpressions",
    "cpm",
)
AD_TYPE_METRICS: tuple[str, ...] = ("grossRevenue", "adImpressions", "cpm")

# insightTrafficSourceDetail is only available for these types. The channel_reports note lists
# VIDEO_REMIXES, NOTIFICATION, END_SCREEN, CAMPAIGN_CARD and NO_LINK_EMBEDDED as unsupported, and
# SHORTS (the swipe feed) has no detail values.
TRAFFIC_DETAIL_TYPES: tuple[str, ...] = (
    "YT_SEARCH",
    "HASHTAGS",
    "SOUND_PAGE",
    "RELATED_VIDEO",
    "YT_CHANNEL",
    "SUBSCRIBER",
    "EXT_URL",
    "YT_OTHER_PAGE",
    "ADVERTISING",
    "WATCH_WITH",
)

DETAIL_MAX_RESULTS = 25  # traffic source detail: maxResults <= 25 and sort is required
CITY_MAX_RESULTS = 250  # user activity by city: maxResults <= 250 and sort is required
CITY_DATA_START = date(2022, 1, 1)  # city data exists from 2022-01-01


@dataclass(frozen=True)
class ReportSpec:
    """One reports.query call (optionally paginated with startIndex/maxResults)."""

    name: str
    metrics: tuple[str, ...]
    dimensions: tuple[str, ...] = ()
    filters: str | None = None
    sort: str | None = None
    max_results: int | None = None
    paginate: bool = False
    table: str | None = None
    min_start: date | None = None

    def params(self, start: date, end: date, start_index: int | None = None) -> dict[str, Any]:
        """Keyword arguments for ``youtubeAnalytics.reports().query``."""
        if self.min_start and start < self.min_start:
            start = self.min_start
        params: dict[str, Any] = {
            "ids": IDS,
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "metrics": ",".join(self.metrics),
        }
        if self.dimensions:
            params["dimensions"] = ",".join(self.dimensions)
        if self.filters:
            params["filters"] = self.filters
        if self.sort:
            params["sort"] = self.sort
        if self.max_results:
            params["maxResults"] = self.max_results
        if start_index is not None:
            params["startIndex"] = start_index
        return params


def run_query(
    analytics: Any,
    caller: ApiCaller,
    spec: ReportSpec,
    start: date,
    end: date,
    raw_name: str,
) -> dict[str, Any]:
    """Execute a report, following startIndex pages when ``spec.paginate``; returns one merged table."""
    if not spec.paginate or not spec.max_results:
        params = spec.params(start, end)
        return caller.execute(
            "analytics", "reports.query", analytics.reports().query(**params), f"{raw_name}.json"
        )

    merged: dict[str, Any] | None = None
    rows: list[list[Any]] = []
    seen: set[tuple[Any, ...]] = set()
    n_dims = len(spec.dimensions)
    start_index = 1
    page = 0
    while True:
        page += 1
        params = spec.params(start, end, start_index=start_index)
        response = caller.execute(
            "analytics",
            "reports.query",
            analytics.reports().query(**params),
            f"{raw_name}.page_{page:03d}.json",
        )
        if merged is None:
            merged = {k: v for k, v in response.items() if k != "rows"}
        page_rows = response.get("rows") or []
        new_rows = [r for r in page_rows if tuple(r[:n_dims]) not in seen]
        for r in new_rows:
            seen.add(tuple(r[:n_dims]))
        rows.extend(new_rows)
        # Stop on a short page, or if the API ignored startIndex and repeated rows.
        if len(page_rows) < spec.max_results or not new_rows:
            break
        start_index += spec.max_results
    assert merged is not None
    if rows:
        merged["rows"] = rows
    return merged


def _with_content_type(
    name: str, table: str, metrics: tuple[str, ...], dims: tuple[str, ...], **kw: Any
) -> list[ReportSpec]:
    """A report both for the whole channel and split by creatorContentType (SHORTS, VIDEO_ON_DEMAND...)."""
    return [
        ReportSpec(name=name, metrics=metrics, dimensions=dims, table=table, **kw),
        ReportSpec(
            name=f"{name}_by_content_type",
            metrics=metrics,
            dimensions=dims + ("creatorContentType",),
            table=table,
            **kw,
        ),
    ]


def channel_report_specs(monetary: bool = False) -> list[ReportSpec]:
    """Channel-wide reports (one call each)."""
    specs: list[ReportSpec] = []
    specs += _with_content_type("channel_daily", "channel_daily", BASIC_METRICS, ("day",))
    specs += _with_content_type("demographics", "demographics", ("viewerPercentage",), ("ageGroup", "gender"))
    specs += _with_content_type("geography", "geography", BASIC_METRICS, ("country",))
    specs += _with_content_type("devices", "devices", INSIGHT_METRICS, ("deviceType", "operatingSystem"))
    specs += _with_content_type(
        "subscribed_status", "subscribed_status", SUBSCRIBED_STATUS_METRICS, ("subscribedStatus",)
    )
    specs += _with_content_type("sharing_services", "sharing_services", ("shares",), ("sharingService",))
    specs += _with_content_type(
        "playback_locations", "playback_locations", INSIGHT_METRICS, ("insightPlaybackLocationType",)
    )
    specs += _with_content_type(
        "channel_traffic_sources", "channel_traffic_sources", INSIGHT_METRICS, ("insightTrafficSourceType",)
    )
    for source in TRAFFIC_DETAIL_TYPES:
        specs.append(
            ReportSpec(
                name=f"channel_traffic_detail_{source}",
                metrics=INSIGHT_METRICS,
                dimensions=("insightTrafficSourceDetail",),
                filters=f"insightTrafficSourceType=={source}",
                sort="-views",
                max_results=DETAIL_MAX_RESULTS,
                table="traffic_source_details",
            )
        )
    specs.append(
        ReportSpec(
            name="cities",
            metrics=VIEW_METRICS,
            dimensions=("city", "country"),
            sort="-views",
            max_results=CITY_MAX_RESULTS,
            table="cities",
            min_start=CITY_DATA_START,
        )
    )
    if monetary:
        specs.append(
            ReportSpec(
                name="revenue_daily", metrics=MONETARY_METRICS, dimensions=("day",), table="revenue_daily"
            )
        )
        specs.append(
            ReportSpec(name="ad_types", metrics=AD_TYPE_METRICS, dimensions=("adType",), table="ad_types")
        )
    return specs
