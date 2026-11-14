"""Raw data loading from nflverse (via nflreadpy), cached to parquet.

Every loader caches under data/raw/. Completed seasons are cached forever; the
current season is re-downloaded when the cache is older than `max_age_hours`.
"""
from __future__ import annotations

import time
from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

FIRST_SEASON = 2013
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def current_season() -> int:
    return nfl.get_current_season()


def _cached(name: str, fetch, max_age_hours: float | None) -> pd.DataFrame:
    path = RAW / f"{name}.parquet"
    fresh = path.exists() and (
        max_age_hours is None or (time.time() - path.stat().st_mtime) < max_age_hours * 3600
    )
    if not fresh:
        df = fetch()
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        df.to_parquet(path, index=False)
    return pd.read_parquet(path)


def _by_season(name: str, loader, seasons: list[int], refresh_hours: float = 12) -> pd.DataFrame:
    """Cache one parquet per season so old seasons never re-download."""
    cur = current_season()
    frames = []
    for s in seasons:
        age = refresh_hours if s >= cur else None
        frames.append(_cached(f"{name}_{s}", lambda s=s: loader([s]), age))
    return pd.concat(frames, ignore_index=True)


def seasons_through_current(first: int = FIRST_SEASON) -> list[int]:
    return list(range(first, current_season() + 1))


def player_stats(seasons=None) -> pd.DataFrame:
    seasons = seasons or seasons_through_current()
    return _by_season("player_stats", nfl.load_player_stats, seasons)


def schedules(seasons=None) -> pd.DataFrame:
    seasons = seasons or seasons_through_current()
    # schedules are small; one file, refreshed like current-season data (lines move)
    return _cached("schedules", lambda: nfl.load_schedules(seasons), 6)


def snap_counts(seasons=None) -> pd.DataFrame:
    seasons = seasons or seasons_through_current()
    return _by_season("snap_counts", nfl.load_snap_counts, seasons)


def ff_opportunity(seasons=None) -> pd.DataFrame:
    seasons = seasons or seasons_through_current()
    out = []
    for s in seasons:
        try:
            out.append(_by_season("ff_opportunity", nfl.load_ff_opportunity, [s]))
        except Exception as e:  # current season can lag; don't fail the pipeline
            print(f"[data] ff_opportunity {s} unavailable: {e}")
    return pd.concat(out, ignore_index=True)


PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "pass",
            "rush", "qb_dropback", "xpass", "pass_oe", "wp", "score_differential", "down", "qtr",
            "air_yards", "yards_gained", "epa", "success", "complete_pass", "receiver_player_id",
            "qb_kneel", "qb_spike", "two_point_attempt"]


def pbp(seasons=None) -> pd.DataFrame:
    """Play-by-play, reduced to the columns the tendency features need
    (full pbp is ~370 columns; cached slim per season)."""
    seasons = seasons or seasons_through_current()
    return _by_season("pbp_slim", lambda s: nfl.load_pbp(s).select(PBP_COLS), seasons)


def injuries(seasons=None) -> pd.DataFrame:
    seasons = seasons or seasons_through_current()
    return _by_season("injuries", nfl.load_injuries, seasons, refresh_hours=3)


def _id_str(s: pd.Series) -> pd.Series:
    """IDs arrive as float or str depending on source; normalize '1234.0' -> '1234'."""
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def player_ids() -> pd.DataFrame:
    df = _cached("ff_playerids", nfl.load_ff_playerids, 24 * 7)
    for c in ["sleeper_id", "fantasypros_id"]:
        df[c] = _id_str(df[c])
    return df


def ecr_history() -> pd.DataFrame:
    """FantasyPros weekly expert consensus rankings archive (2020+)."""
    return _cached("ff_rankings_all", lambda: nfl.load_ff_rankings("all"), 24 * 3)


def ecr_this_week() -> pd.DataFrame:
    return _cached("ff_rankings_week", lambda: nfl.load_ff_rankings("week"), 6)
