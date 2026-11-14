"""Feature engineering.

Design: build a per-(player, game) history table with *inclusive* rolling stats,
then attach to each prediction row (player, season, week) the most recent
history row strictly before it via `merge_asof(allow_exact_matches=False)`.
That one mechanism gives leakage-free features for backtest rows AND for
future weeks (where the target is unknown), so train and serve share code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .scoring import fantasy_points

SKILL = data.SKILL_POSITIONS
POS_CODE = {p: i for i, p in enumerate(SKILL)}
RELOCATED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

USAGE_COLS = [
    "pts", "xfp", "targets", "receptions", "carries", "attempts",
    "target_share", "air_yards_share", "wopr", "receiving_air_yards",
    "rushing_yards", "receiving_yards", "passing_yards", "offense_pct",
    "tds",
]


def _tkey(season, week):
    return season.astype(int) * 100 + week.astype(int)


# --------------------------------------------------------------------- tables
def team_games() -> pd.DataFrame:
    """One row per team per regular-season game, with Vegas/game context."""
    s = data.schedules()
    s = s[s["game_type"] == "REG"].copy()
    for c in ["home_team", "away_team"]:  # player_stats uses current franchise codes
        s[c] = s[c].replace(RELOCATED)
    s["completed"] = s["home_score"].notna()
    common = ["game_id", "season", "week", "gameday", "total_line", "roof", "wind", "temp",
              "div_game", "completed"]
    home = s[common].assign(
        team=s["home_team"], opp=s["away_team"], home=1,
        spread=s["spread_line"], rest=s["home_rest"], qb_id=s["home_qb_id"],
    )
    away = s[common].assign(
        team=s["away_team"], opp=s["home_team"], home=0,
        spread=-s["spread_line"], rest=s["away_rest"], qb_id=s["away_qb_id"],
    )
    g = pd.concat([home, away], ignore_index=True).sort_values(["team", "season", "week"])
    # starters aren't listed beyond the next week: carry the last known starter forward
    g["qb_id"] = g.groupby(["team", "season"])["qb_id"].ffill()
    prev = g.groupby(["team", "season"])["qb_id"].shift()
    g["qb_changed"] = (prev.notna() & (prev != g["qb_id"])).astype(int)
    # nflverse spread_line > 0 means home favored, so team expected margin = spread
    g["implied_total"] = g["total_line"] / 2 + g["spread"] / 2
    g["dome"] = g["roof"].isin(["dome", "closed"]).astype(int)
    g["wind"] = g["wind"].where(g["dome"] == 0, 0)
    g = _impute_future_lines(g)
    return g.drop(columns=["roof"])


def _impute_future_lines(g: pd.DataFrame) -> pd.DataFrame:
    """Sportsbooks only post ~1 week ahead. For later weeks (rest-of-season
    projections) estimate the line from both teams' season-average lines."""
    g = g.copy()
    g["line_imputed"] = g["total_line"].isna().astype(int)
    known = g[g["total_line"].notna()]
    avg = known.groupby(["season", "team"]).agg(a_tot=("total_line", "mean"), a_spr=("spread", "mean"))
    m = g["total_line"].isna()
    if m.any():
        x = g[m].join(avg, on=["season", "team"]).join(
            avg.rename(columns={"a_tot": "o_tot", "a_spr": "o_spr"}), on=["season", "opp"])
        g.loc[m, "total_line"] = ((x["a_tot"] + x["o_tot"]) / 2).values
        g.loc[m, "spread"] = ((x["a_spr"] - x["o_spr"]) / 2 + np.where(x["home"] == 1, 1.5, -1.5)).values
        g.loc[m, "implied_total"] = g.loc[m, "total_line"] / 2 + g.loc[m, "spread"] / 2
    return g


def game_table(scoring: dict | None = None) -> pd.DataFrame:
    """Completed regular-season skill-position games with fantasy points + usage."""
    ps = data.player_stats()
    ps = ps[(ps["season_type"] == "REG") & ps["position"].isin(SKILL)].copy()
    ps["pts"] = fantasy_points(ps, scoring)

    ids = data.player_ids()
    pfr = ids[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    sn = data.snap_counts()
    sn = sn[sn["game_type"] == "REG"].merge(pfr, left_on="pfr_player_id", right_on="pfr_id")
    sn = (sn.rename(columns={"gsis_id": "player_id"})[["player_id", "season", "week", "offense_pct"]]
            .drop_duplicates(["player_id", "season", "week"]))
    ps = ps.merge(sn, on=["player_id", "season", "week"], how="left")

    op = data.ff_opportunity()
    op = op.assign(season=op["season"].astype(int), week=op["week"].astype(int))
    op = op[["player_id", "season", "week", "total_fantasy_points_exp", "rec_attempt", "rush_attempt"]]
    op = op.rename(columns={"total_fantasy_points_exp": "xfp"}).drop_duplicates(["player_id", "season", "week"])
    ps = ps.merge(op[["player_id", "season", "week", "xfp"]], on=["player_id", "season", "week"], how="left")

    # crude red-zone proxy unavailable in weekly stats; use TD-adjacent volume
    ps["tds"] = ps[["rushing_tds", "receiving_tds"]].fillna(0).sum(axis=1)

    meta = ids[["gsis_id", "draft_year", "draft_ovr", "sleeper_id", "fantasypros_id"]].dropna(subset=["gsis_id"])
    meta = meta.drop_duplicates("gsis_id").rename(columns={"gsis_id": "player_id"})
    ps = ps.merge(meta, on="player_id", how="left")
    ps["t"] = _tkey(ps["season"], ps["week"])
    return ps.sort_values(["player_id", "t"]).reset_index(drop=True)


# ------------------------------------------------------------ rolling history
def player_history(games: pd.DataFrame) -> pd.DataFrame:
    """Inclusive rolling stats as of the end of each game (no shift here:
    the as-of join with allow_exact_matches=False provides the lag)."""
    g = games
    out = g[["player_id", "t", "season", "week", "team"]].copy()
    grp = g.groupby("player_id", sort=False)
    for c in USAGE_COLS:
        x = g[c].astype(float)
        out[f"{c}_l3"] = grp[c].transform(lambda s: s.astype(float).rolling(3, min_periods=1).mean())
        out[f"{c}_l8"] = grp[c].transform(lambda s: s.astype(float).rolling(8, min_periods=1).mean())
        out[f"{c}_ewm"] = grp[c].transform(lambda s: s.astype(float).ewm(halflife=3, ignore_na=True).mean())
        out[f"{c}_last"] = x
    out["pts_std8"] = grp["pts"].transform(lambda s: s.rolling(8, min_periods=2).std())
    out["pts_max8"] = grp["pts"].transform(lambda s: s.rolling(8, min_periods=1).max())
    out["n_games"] = grp.cumcount() + 1
    sgrp = g.groupby(["player_id", "season"], sort=False)
    out["pts_std_season"] = sgrp["pts"].transform(lambda s: s.expanding().mean())
    out["xfp_std_season"] = sgrp["xfp"].transform(lambda s: s.astype(float).expanding().mean())
    out["games_season"] = sgrp.cumcount() + 1
    out = out.rename(columns={"season": "h_season", "week": "h_week", "team": "h_team"})
    return out


def defense_history(games: pd.DataFrame) -> pd.DataFrame:
    """Points allowed by each defense to each position, rolling."""
    a = (games.groupby(["opponent_team", "position", "season", "week"], as_index=False)["pts"].sum()
              .rename(columns={"opponent_team": "opp"}))
    a["t"] = _tkey(a["season"], a["week"])
    a = a.sort_values(["opp", "position", "t"])
    grp = a.groupby(["opp", "position"], sort=False)["pts"]
    a["def_allowed_l8"] = grp.transform(lambda s: s.rolling(8, min_periods=1).mean())
    a["def_allowed_ewm"] = grp.transform(lambda s: s.ewm(halflife=4).mean())
    return a[["opp", "position", "t", "def_allowed_l8", "def_allowed_ewm"]]


# ------------------------------------------------------------ assembly
def prediction_rows(games: pd.DataFrame, include_future: bool = False,
                    current_teams: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows to predict: every completed game (with target) plus, optionally,
    every not-yet-played game this season for each player active this season.

    `current_teams` (player_id, team) — e.g. from Sleeper — overrides the
    nflverse team (catches trades/signings) and adds players with history but
    no game yet this season (e.g. returning from injury)."""
    tg = team_games()
    hist = games[["player_id", "player_display_name", "position", "season", "week", "team",
                  "pts", "draft_year", "draft_ovr", "sleeper_id", "fantasypros_id"]].copy()
    hist = hist.merge(tg, on=["season", "week", "team"], how="left")
    rows = [hist]

    if include_future:
        season = data.current_season()
        cur = games[games["season"] == season].sort_values("t")
        latest = cur.groupby("player_id").tail(1)[[
            "player_id", "player_display_name", "position", "team",
            "draft_year", "draft_ovr", "sleeper_id", "fantasypros_id"]]
        if current_teams is not None and len(current_teams):
            ct = current_teams.dropna().drop_duplicates("player_id").set_index("player_id")["team"]
            missing = ct.index.difference(latest["player_id"])
            # only players who played this or last season (Sleeper keeps some
            # retired players flagged Active on their old team)
            recent = games["season"] >= season - 1
            prior = (games[games["player_id"].isin(missing) & recent].sort_values("t")
                     .groupby("player_id").tail(1)[latest.columns])
            latest = pd.concat([latest, prior], ignore_index=True)
            latest["team"] = latest["player_id"].map(ct).fillna(latest["team"])
        fut_tg = tg[(tg["season"] == season) & ~tg["completed"]]
        fut = latest.merge(fut_tg, on="team", how="inner")
        fut["pts"] = np.nan
        rows.append(fut)

    r = pd.concat(rows, ignore_index=True)
    r["t"] = _tkey(r["season"], r["week"])
    return r


STATUS_CODE = {"Questionable": 1, "Doubtful": 2, "Out": 3}
VACATED_COLS = ["targets_ewm", "carries_ewm", "xfp_ewm"]


def injury_features(ph: pd.DataFrame, overrides: pd.DataFrame | None = None):
    """Official pre-game injury reports -> (own status per player-week,
    usage vacated by teammates ruled Out/Doubtful per team-week[-position]).

    Only teammates who played for that team within the last 2 weeks count, so
    long-term IR absences (already priced into teammates' recent usage) don't.

    `overrides` (player_id, season, week, team, position, status_code) adds
    fresher statuses — e.g. Sleeper's, which update after the official report
    (a Saturday 'won't play' ruling). The more severe status wins."""
    inj = data.injuries()
    inj = inj[(inj["game_type"] == "REG") & inj["report_status"].isin(STATUS_CODE)].copy()
    inj["team"] = inj["team"].replace(RELOCATED)
    inj["status_code"] = inj["report_status"].map(STATUS_CODE)
    inj = inj.rename(columns={"gsis_id": "player_id"})
    if overrides is not None and len(overrides):
        inj = pd.concat([inj, overrides], ignore_index=True)
    inj["t"] = _tkey(inj["season"], inj["week"])
    inj = (inj.sort_values("status_code").drop_duplicates(["player_id", "season", "week"], keep="last")
              [["player_id", "season", "week", "t", "team", "position", "status_code"]])

    own = inj[["player_id", "season", "week", "status_code"]].rename(columns={"status_code": "own_status"})

    out = inj[inj["status_code"] >= 2].sort_values("t")
    out = pd.merge_asof(out, ph[["player_id", "t", "h_season", "h_week", "h_team"] + VACATED_COLS],
                        on="t", by="player_id", allow_exact_matches=False)
    recent = (out["h_season"] == out["season"]) & (out["week"] - out["h_week"] <= 2) & (out["h_team"] == out["team"])
    out = out[recent].fillna({c: 0 for c in VACATED_COLS})
    team_vac = (out.groupby(["season", "week", "team"])[VACATED_COLS].sum()
                   .add_prefix("team_vacated_").reset_index())
    pos_vac = (out.groupby(["season", "week", "team", "position"])["xfp_ewm"].sum()
                  .rename("pos_vacated_xfp").reset_index())
    return own, team_vac, pos_vac


def qb_features(games: pd.DataFrame, tg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(QB quality history keyed by qb_id, player-with-this-QB history keyed by
    (player_id, qb_id)). Both inclusive; lagged by the as-of join."""
    qb = games[games["position"] == "QB"].copy()
    qb["epa_db"] = qb["passing_epa"] / (qb["attempts"] + qb["sacks_suffered"].fillna(0)).replace(0, np.nan)
    qb = qb.sort_values(["player_id", "t"])
    g = qb.groupby("player_id", sort=False)
    qh = pd.DataFrame({
        "qb_id": qb["player_id"], "t": qb["t"],
        "qb_pts_ewm": g["pts"].transform(lambda s: s.ewm(halflife=4).mean()),
        "qb_epa_ewm": g["epa_db"].transform(lambda s: s.ewm(halflife=4, ignore_na=True).mean()),
        "qb_n_games": g.cumcount() + 1,
    })
    gq = games.merge(tg[["season", "week", "team", "qb_id"]], on=["season", "week", "team"], how="left")
    gq["qb_id"] = gq["qb_id"].fillna("UNK")
    gq = gq.sort_values(["player_id", "qb_id", "t"])
    g2 = gq.groupby(["player_id", "qb_id"], sort=False)
    wq = pd.DataFrame({
        "player_id": gq["player_id"], "qb_id": gq["qb_id"], "t": gq["t"],
        "wq_target_share_ewm": g2["target_share"].transform(lambda s: s.astype(float).ewm(halflife=3, ignore_na=True).mean()),
        "wq_pts_ewm": g2["pts"].transform(lambda s: s.ewm(halflife=3).mean()),
        "wq_games": g2.cumcount() + 1,
    })
    return qh.sort_values("t"), wq.sort_values("t")


SHORT_AY = 10  # air yards; "short" ≈ slot/underneath targets


def team_tendencies() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coordinator fingerprints from play-by-play, per team-game, rolled (EWM):

    offense (posteam): plays/game, neutral-situation pass rate, pass rate over
      expected (PROE), RB share of targets, short-target share.
    defense (defteam): yards & EPA per target allowed on short vs deep targets,
      share of targets faced that are short, rush EPA allowed.
    Inclusive rolling values; the as-of join lags them."""
    p = data.pbp()
    p = p[(p["season_type"] == "REG") & p["play_type"].isin(["pass", "run"])
          & (p["qb_kneel"] != 1) & (p["qb_spike"] != 1) & (p["two_point_attempt"] != 1)].copy()
    for c in ["posteam", "defteam"]:
        p[c] = p[c].replace(RELOCATED)
    p["neutral"] = (p["wp"].between(0.2, 0.8) & p["down"].isin([1, 2]) & (p["qtr"] <= 3))
    tgt = p[(p["pass"] == 1) & p["air_yards"].notna()].copy()
    tgt["short"] = tgt["air_yards"] < SHORT_AY

    key = ["season", "week"]
    off = p.groupby(["posteam"] + key).agg(
        plays=("play_type", "size"),
        pass_rate_neutral=("pass", lambda s: s[p.loc[s.index, "neutral"]].mean()),
        proe=("pass_oe", "mean"),
    )
    off = off.join(tgt.groupby(["posteam"] + key)["short"].mean().rename("short_tgt_share"))
    off = off.reset_index().rename(columns={"posteam": "team"})

    d_short = tgt[tgt["short"]].groupby(["defteam"] + key).agg(
        def_short_ypt=("yards_gained", "mean"), def_short_epa=("epa", "mean"))
    d_deep = tgt[~tgt["short"]].groupby(["defteam"] + key).agg(
        def_deep_ypt=("yards_gained", "mean"), def_deep_epa=("epa", "mean"))
    d_mix = tgt.groupby(["defteam"] + key)["short"].mean().rename("def_short_share_faced")
    d_rush = p[p["rush"] == 1].groupby(["defteam"] + key)["epa"].mean().rename("def_rush_epa")
    dfn = d_short.join([d_deep, d_mix, d_rush], how="outer").reset_index().rename(columns={"defteam": "opp"})

    def roll(df, by, cols):
        df = df.copy()
        df["t"] = _tkey(df["season"], df["week"])
        df = df.sort_values([by, "t"])
        for c in cols:
            df[f"{c}_ewm"] = df.groupby(by)[c].transform(lambda s: s.ewm(halflife=4, ignore_na=True).mean())
        return df[[by, "t"] + [f"{c}_ewm" for c in cols]].sort_values("t")

    off_cols = ["plays", "pass_rate_neutral", "proe", "short_tgt_share"]
    def_cols = ["def_short_ypt", "def_short_epa", "def_deep_ypt", "def_deep_epa",
                "def_short_share_faced", "def_rush_epa"]
    return roll(off, "team", off_cols), roll(dfn, "opp", def_cols)


TENDENCY_COLS = (["plays_ewm", "pass_rate_neutral_ewm", "proe_ewm", "short_tgt_share_ewm"]
                 + [f"{c}_ewm" for c in ["def_short_ypt", "def_short_epa", "def_deep_ypt", "def_deep_epa",
                                         "def_short_share_faced", "def_rush_epa"]]
                 + ["adot_ewm", "def_depth_match_ypt"])


CORE_SHARE = 0.12  # a "core" pass-catcher: recent target share >= 12%


def role_features(rows: pd.DataFrame, games: pd.DataFrame, ph: pd.DataFrame,
                  overrides: pd.DataFrame | None) -> pd.DataFrame:
    """Usage *in the current role*: each player's usage over earlier games
    where the same core teammates were missing.

    A team-game's 'regime' = the set of core teammates (same team & season,
    recent target share >= CORE_SHARE) who are absent. Completed games use who
    actually played (inactives are announced pre-kickoff). Future games use
    statuses: Out/Doubtful this week, and players already out 2+ straight
    games are assumed to stay out (e.g. WR1 and WR2 both injured for weeks ->
    the WR3's elevated role persists). Returns per-row features."""
    skill = games[games["position"].isin(["RB", "WR", "TE"])]
    member = skill[["player_id", "team", "season"]].drop_duplicates()
    tg = rows[["season", "week", "team", "t"]].drop_duplicates()
    cand = tg.merge(member, on=["team", "season"])
    hist = ph[["player_id", "t", "h_season", "h_week", "h_team", "target_share_ewm"]].sort_values("t")
    cand = pd.merge_asof(cand.sort_values("t"), hist, on="t", by="player_id", allow_exact_matches=False)
    core = cand[(cand["h_season"] == cand["season"]) & (cand["h_team"] == cand["team"])
                & (cand["target_share_ewm"] >= CORE_SHARE)].copy()

    played = set(zip(games["player_id"], games["season"], games["week"]))
    done = set(zip(games["season"], games["week"], games["team"]))
    is_done = [(s, w, tm) in done for s, w, tm in zip(core["season"], core["week"], core["team"])]
    core["completed"] = is_done
    core["absent"] = [(p, s, w) not in played for p, s, w in zip(core["player_id"], core["season"], core["week"])]

    # future games: absence from statuses
    fut = ~core["completed"]
    if fut.any():
        out_now = set()
        if overrides is not None and len(overrides):
            o = overrides[overrides["status_code"] >= 2]
            out_now = set(o["player_id"])
        cur_week = core.loc[fut, "week"].min()
        # already sat out the last 2+ team weeks -> assume the absence persists, with or
        # without an injury tag (healthy scratches like a demoted TE carry no status).
        # Trade-off: a player returning from a long absence with no tag also looks absent.
        missed = (cur_week - core["h_week"]) >= 3
        core.loc[fut, "absent"] = missed[fut] | (
            core.loc[fut, "player_id"].isin(out_now) & (core.loc[fut, "week"] == cur_week))

    ab = core[core["absent"]]
    key = (ab.groupby(["season", "week", "team"])
             .agg(rg_key=("player_id", lambda s: "|".join(sorted(s))),
                  n_core_absent=("player_id", "size"),
                  core_absent_share=("target_share_ewm", "sum"))
             .reset_index())
    tg = tg.merge(key, on=["season", "week", "team"], how="left")
    tg["rg_key"] = tg["rg_key"].fillna("none")
    tg[["n_core_absent", "core_absent_share"]] = tg[["n_core_absent", "core_absent_share"]].fillna(0)

    # usage history keyed by (player, season, regime): inclusive EWMs, lagged by as-of join
    g = games.merge(tg[["season", "week", "team", "rg_key"]], on=["season", "week", "team"], how="left")
    g["rg_key"] = g["rg_key"].fillna("none")
    g = g.sort_values(["player_id", "season", "rg_key", "t"])
    grp = g.groupby(["player_id", "season", "rg_key"], sort=False)
    rh = pd.DataFrame({
        "player_id": g["player_id"], "season_rg": g["season"], "rg_key": g["rg_key"], "t": g["t"],
        "rg_target_share_ewm": grp["target_share"].transform(lambda s: s.astype(float).ewm(halflife=3, ignore_na=True).mean()),
        "rg_offense_pct_ewm": grp["offense_pct"].transform(lambda s: s.astype(float).ewm(halflife=3, ignore_na=True).mean()),
        "rg_pts_ewm": grp["pts"].transform(lambda s: s.ewm(halflife=3).mean()),
        "rg_games": grp.cumcount() + 1,
    }).sort_values("t")
    out = rows[["player_id", "season", "week", "team", "t"]].merge(
        tg[["season", "week", "team", "rg_key", "n_core_absent", "core_absent_share"]],
        on=["season", "week", "team"], how="left")
    out["rg_key"] = out["rg_key"].fillna("none")
    out["season_rg"] = out["season"]
    out = pd.merge_asof(out.sort_values("t"), rh, on="t", by=["player_id", "season_rg", "rg_key"],
                        allow_exact_matches=False)
    out["rg_games"] = out["rg_games"].fillna(0)
    return out[["player_id", "season", "week", "rg_target_share_ewm", "rg_offense_pct_ewm",
                "rg_pts_ewm", "rg_games", "n_core_absent", "core_absent_share"]]


ROLE_COLS = ["rg_target_share_ewm", "rg_offense_pct_ewm", "rg_pts_ewm", "rg_games",
             "n_core_absent", "core_absent_share", "rg_share_diff"]
USE_ROLE = False  # tested 2026-11-14: WR/TE MAE slightly better, live-blend ranking slightly worse (see README)


def attach_features(rows: pd.DataFrame, games: pd.DataFrame,
                    injury_overrides: pd.DataFrame | None = None) -> pd.DataFrame:
    ph = player_history(games).sort_values("t")
    dh = defense_history(games).sort_values("t")
    r = rows.sort_values("t").reset_index(drop=True)

    r = pd.merge_asof(r, ph, on="t", by="player_id", allow_exact_matches=False)
    r = pd.merge_asof(r.sort_values("t"), dh, on="t", by=["opp", "position"], allow_exact_matches=False)

    # starting QB: who he is, how good, whether he's new, and this player's usage *with him*
    qh, wq = qb_features(games, team_games())
    r["qb_id"] = r["qb_id"].fillna("UNK")
    r = pd.merge_asof(r.sort_values("t"), qh, on="t", by="qb_id", allow_exact_matches=False)
    r = pd.merge_asof(r.sort_values("t"), wq, on="t", by=["player_id", "qb_id"], allow_exact_matches=False)
    r["wq_games"] = r["wq_games"].fillna(0)
    r["qb_n_games"] = r["qb_n_games"].fillna(0)
    r["wq_share_diff"] = r["wq_target_share_ewm"] - r["target_share_ewm"]

    # usage in the current role (same core teammates missing)
    if USE_ROLE:
        rf = role_features(r, games, ph, injury_overrides)
        r = r.merge(rf, on=["player_id", "season", "week"], how="left")
        r["rg_share_diff"] = r["rg_target_share_ewm"] - r["target_share_ewm"]

    # offensive / defensive coordinator tendencies
    if USE_TENDENCIES:
        offh, defh = team_tendencies()
        r = pd.merge_asof(r.sort_values("t"), offh, on="t", by="team", allow_exact_matches=False)
        r = pd.merge_asof(r.sort_values("t"), defh, on="t", by="opp", allow_exact_matches=False)
        # player's route depth, and what this defense allows at that depth
        r["adot_ewm"] = r["receiving_air_yards_ewm"] / r["targets_ewm"].where(r["targets_ewm"] > 0)
        short_p = (r["adot_ewm"] < SHORT_AY).astype(float).where(r["adot_ewm"].notna())
        r["def_depth_match_ypt"] = np.where(short_p == 1, r["def_short_ypt_ewm"],
                                   np.where(short_p == 0, r["def_deep_ypt_ewm"], np.nan))

    own, team_vac, pos_vac = injury_features(ph, injury_overrides)
    r = r.merge(own, on=["player_id", "season", "week"], how="left")
    r = r.merge(team_vac, on=["season", "week", "team"], how="left")
    r = r.merge(pos_vac, on=["season", "week", "team", "position"], how="left")
    inj_cols = ["own_status", "pos_vacated_xfp"] + [f"team_vacated_{c}" for c in VACATED_COLS]
    r[inj_cols] = r[inj_cols].fillna(0)

    same_season = r["h_season"] == r["season"]
    for c in ["pts_std_season", "xfp_std_season", "games_season"]:
        r[c] = r[c].where(same_season)
    r["games_season"] = r["games_season"].fillna(0)
    r["weeks_since_last"] = (r["week"] - r["h_week"]).where(same_season)
    r["new_team"] = ((r["h_team"] != r["team"]) & r["h_team"].notna()).astype(int)
    r["years_exp"] = (r["season"] - r["draft_year"]).clip(lower=0)
    r["draft_ovr"] = r["draft_ovr"].fillna(300)
    r["pos_code"] = r["position"].map(POS_CODE)
    r["n_games"] = r["n_games"].fillna(0)
    return r


def feature_columns() -> list[str]:
    cols = []
    for c in USAGE_COLS:
        cols += [f"{c}_l3", f"{c}_l8", f"{c}_ewm", f"{c}_last"]
    cols += [
        "pts_std8", "pts_max8", "n_games", "pts_std_season", "xfp_std_season", "games_season",
        "weeks_since_last", "new_team", "years_exp", "draft_ovr", "pos_code", "week",
        "home", "spread", "total_line", "implied_total", "rest", "dome", "wind", "temp", "div_game",
        "def_allowed_l8", "def_allowed_ewm",
        "own_status", "pos_vacated_xfp",
    ] + [f"team_vacated_{c}" for c in VACATED_COLS] + QB_COLS
    if USE_TENDENCIES:
        cols += TENDENCY_COLS
    if USE_ROLE:
        cols += ROLE_COLS
    return cols


# coordinator-tendency features (pbp-derived); toggled for the ablation
USE_TENDENCIES = False  # tested 2026-11-14: no gain (see README ablation)


QB_COLS = ["qb_changed", "qb_pts_ewm", "qb_epa_ewm", "qb_n_games",
           "wq_target_share_ewm", "wq_pts_ewm", "wq_games", "wq_share_diff"]


def build_dataset(scoring: dict | None = None, include_future: bool = False,
                  current_teams: pd.DataFrame | None = None,
                  injury_overrides: pd.DataFrame | None = None):
    games = game_table(scoring)
    rows = prediction_rows(games, include_future, current_teams)
    return attach_features(rows, games, injury_overrides), games
