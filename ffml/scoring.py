"""Fantasy scoring. Keys follow Sleeper's `scoring_settings` names so a league's
settings dict can be passed straight in."""
from __future__ import annotations

import pandas as pd

HALF_PPR = {
    "pass_yd": 0.04, "pass_td": 4, "pass_int": -1, "pass_2pt": 2,
    "rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2,
    "rec": 0.5, "rec_yd": 0.1, "rec_td": 6, "rec_2pt": 2,
    "fum_lost": -2, "st_td": 6,
    "bonus_rec_te": 0.0, "rec_fd": 0.0, "rush_fd": 0.0,
}

# Sleeper key -> function(player_stats df) -> Series
_COMPONENTS = {
    "pass_yd": lambda d: d["passing_yards"],
    "pass_td": lambda d: d["passing_tds"],
    "pass_int": lambda d: d["passing_interceptions"],
    "pass_2pt": lambda d: d["passing_2pt_conversions"],
    "rush_yd": lambda d: d["rushing_yards"],
    "rush_td": lambda d: d["rushing_tds"],
    "rush_2pt": lambda d: d["rushing_2pt_conversions"],
    "rec": lambda d: d["receptions"],
    "rec_yd": lambda d: d["receiving_yards"],
    "rec_td": lambda d: d["receiving_tds"],
    "rec_2pt": lambda d: d["receiving_2pt_conversions"],
    "fum_lost": lambda d: d["rushing_fumbles_lost"] + d["receiving_fumbles_lost"] + d["sack_fumbles_lost"],
    "st_td": lambda d: d["special_teams_tds"],
    "bonus_rec_te": lambda d: d["receptions"] * (d["position"] == "TE"),
    "rec_fd": lambda d: d["receiving_first_downs"],
    "rush_fd": lambda d: d["rushing_first_downs"],
}


def fantasy_points(stats: pd.DataFrame, settings: dict | None = None) -> pd.Series:
    settings = {**HALF_PPR, **(settings or {})}
    pts = pd.Series(0.0, index=stats.index)
    for key, fn in _COMPONENTS.items():
        w = settings.get(key, 0) or 0
        if w:
            pts = pts + w * fn(stats).fillna(0).astype(float)
    return pts


def differs_from_half_ppr(settings: dict) -> dict:
    """Return the scoring keys (that this model knows about) where a league
    deviates from half-PPR — the model was trained on half-PPR."""
    out = {}
    for k, v in HALF_PPR.items():
        if k not in settings:
            continue
        lv = settings.get(k) or 0
        if abs(lv - v) > 1e-9:
            out[k] = (v, lv)
    return out
