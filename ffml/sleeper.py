"""Read-only Sleeper API client (public, no auth). https://docs.sleeper.com"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.sleeper.app/v1"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
_S = requests.Session()
_S.headers["User-Agent"] = "ffml-fantasy-model/0.1"


def _get(path: str):
    r = _S.get(f"{BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def state() -> dict:
    return _get("/state/nfl")


def user(username: str) -> dict:
    u = _get(f"/user/{username}")
    if not u:
        raise SystemExit(f"Sleeper user '{username}' not found")
    return u


def leagues(user_id: str, season: int | str) -> list[dict]:
    return _get(f"/user/{user_id}/leagues/nfl/{season}")


def league(league_id: str) -> dict:
    return _get(f"/league/{league_id}")


def rosters(league_id: str) -> list[dict]:
    return _get(f"/league/{league_id}/rosters")


def league_users(league_id: str) -> list[dict]:
    return _get(f"/league/{league_id}/users")


def trending_adds(hours: int = 48, limit: int = 100) -> dict[str, int]:
    rows = _get(f"/players/nfl/trending/add?lookback_hours={hours}&limit={limit}")
    return {r["player_id"]: r["count"] for r in rows}


def players(max_age_hours: float = 24) -> pd.DataFrame:
    """All NFL players (~5 MB). Sleeper asks clients to fetch this at most daily."""
    p = RAW / "sleeper_players.json"
    if not p.exists() or time.time() - p.stat().st_mtime > max_age_hours * 3600:
        p.write_text(json.dumps(_get("/players/nfl")))
    d = json.loads(p.read_text())
    df = pd.DataFrame.from_dict(d, orient="index")
    df.index.name = "sleeper_id"
    df = df.reset_index()
    keep = ["sleeper_id", "full_name", "first_name", "last_name", "position", "team", "status",
            "active", "injury_status", "gsis_id", "fantasy_positions", "search_rank"]
    df = df[[c for c in keep if c in df.columns]]
    df["gsis_id"] = df["gsis_id"].astype("string").str.strip()
    # team defenses have id == team abbreviation and no names
    is_def = df["position"] == "DEF"
    df.loc[is_def, "full_name"] = df.loc[is_def, "sleeper_id"] + " D/ST"
    df.loc[is_def, "team"] = df.loc[is_def, "sleeper_id"]
    return df


def resolve_league(username: str, season: int, league_id: str | None = None) -> tuple[dict, dict]:
    """Return (league, my_roster). If the user is in several leagues and no
    league_id is given, list them and exit so the user can pick."""
    u = user(username)
    if league_id is None:
        ls = leagues(u["user_id"], season)
        if not ls:
            raise SystemExit(f"No {season} NFL leagues for {username}")
        if len(ls) > 1:
            lines = "\n".join(f"  {l['league_id']}  {l['name']}" for l in ls)
            raise SystemExit(f"{username} is in {len(ls)} leagues; pass --league <id>:\n{lines}")
        league_id = ls[0]["league_id"]
    lg = league(league_id)
    mine = [r for r in rosters(league_id) if r.get("owner_id") == u["user_id"]
            or u["user_id"] in (r.get("co_owners") or [])]
    if not mine:
        raise SystemExit(f"{username} has no roster in league {league_id}")
    return lg, mine[0]


def rostered_ids(league_id: str) -> set[str]:
    out = set()
    for r in rosters(league_id):
        out.update(r.get("players") or [])
    return out


def historical_projections(season: int, week: int, max_age_hours: float | None = None) -> pd.DataFrame:
    """Sleeper's weekly projections (sourced from Rotowire) via the unofficial
    api.sleeper.com endpoint. Cached per week; completed weeks never refetch.
    Pass max_age_hours for current/future weeks, whose projections still move."""
    p = RAW / "sleeper_proj" / f"{season}_{week:02d}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and (max_age_hours is None or time.time() - p.stat().st_mtime < max_age_hours * 3600):
        return pd.read_parquet(p)
    qs = "&".join(f"position%5B%5D={x}" for x in ["QB", "RB", "WR", "TE"])
    r = _S.get(f"https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular&{qs}", timeout=30)
    r.raise_for_status()
    rows = [{"sleeper_id": x["player_id"], "season": int(x["season"]), "week": int(x["week"]),
             "sleeper_proj": (x.get("stats") or {}).get("pts_half_ppr"),
             "company": x.get("company"), "last_modified": x.get("last_modified"),
             "date": x.get("date")} for x in r.json()]
    df = pd.DataFrame(rows)
    time.sleep(0.3)
    df.to_parquet(p, index=False)
    return df
