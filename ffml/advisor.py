"""Start/sit and waiver recommendations for a Sleeper roster."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, features, model, sleeper
from .ecr import ecr_to_points

MODELED = {"QB", "RB", "WR", "TE"}
SLOT_ELIGIBLE = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "K": {"K"}, "DEF": {"DEF"},
    "FLEX": {"RB", "WR", "TE"}, "WRRB_FLEX": {"RB", "WR"}, "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}
OUT_THIS_WEEK = {"Out", "IR", "PUP", "Sus", "NA", "DNR"}
OUT_LONG_TERM = {"IR", "PUP", "Sus", "NA", "DNR"}
# weight on the model (vs FantasyPros ECR) per position, chosen by the
# walk-forward backtest (reports/blend_weights.md). Experts are measurably
# better at WR; elsewhere model and ECR tie and the blend edges both.
MODEL_WEIGHT = {"QB": 0.5, "RB": 0.25, "WR": 0.25, "TE": 0.5}
SLEEPER_TO_NFLVERSE = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV"}


# ------------------------------------------------------------------ lineup
def optimal_lineup(pool: pd.DataFrame, slots: list[str], score: str = "score") -> tuple[list, float]:
    """Greedy fill, most restrictive slots first. Exact for the nested
    eligibility sets fantasy leagues use (QB ⊂ SUPER_FLEX, RB ⊂ FLEX, ...)."""
    slots = sorted([s for s in slots if s in SLOT_ELIGIBLE], key=lambda s: len(SLOT_ELIGIBLE[s]))
    avail = pool.sort_values(score, ascending=False)
    used, lineup, total = set(), [], 0.0
    for s in slots:
        cand = avail[avail["position"].isin(SLOT_ELIGIBLE[s]) & ~avail["sleeper_id"].isin(used)]
        if cand.empty:
            lineup.append((s, None)); continue
        best = cand.iloc[0]
        used.add(best["sleeper_id"]); lineup.append((s, best["sleeper_id"]))
        total += float(best[score])
    return lineup, total


# ------------------------------------------------------------------ projections
def project(sp: pd.DataFrame, models: dict, season: int, week: int) -> pd.DataFrame:
    """Projections for every remaining game this season, keyed by sleeper_id."""
    ids = data.player_ids()[["sleeper_id", "gsis_id"]].dropna().drop_duplicates("sleeper_id")
    sp = sp.merge(ids.rename(columns={"gsis_id": "gsis_xwalk"}), on="sleeper_id", how="left")
    sp["player_id"] = sp["gsis_xwalk"].fillna(sp["gsis_id"])
    sp["nfl_team"] = sp["team"].replace(SLEEPER_TO_NFLVERSE)
    skill = sp[sp["position"].isin(MODELED) & sp["player_id"].notna()]
    cur = skill[["player_id", "nfl_team"]].rename(columns={"nfl_team": "team"}).dropna()

    # Sleeper's injury status updates after the official report (e.g. a Saturday
    # 'won't play'); feed it in for this week so teammates' vacated usage counts.
    sev = {"Questionable": 1, "Doubtful": 2, "Out": 3, "IR": 3, "PUP": 3, "Sus": 3}
    ov = skill[skill["injury_status"].isin(sev)].assign(
        season=season, week=week, status_code=lambda d: d["injury_status"].map(sev))
    ov = ov.rename(columns={"nfl_team": "team_ov"})[["player_id", "season", "week", "team_ov", "position", "status_code"]]
    ov = ov.rename(columns={"team_ov": "team"})

    df, _ = features.build_dataset(include_future=True, current_teams=cur, injury_overrides=ov)
    fut = df[(df["season"] == season) & (df["pts"].isna()) & (df["week"] >= week)]
    p = model.predict(models, fut)
    p = p.merge(skill[["player_id", "sleeper_id"]].rename(columns={"sleeper_id": "sid"}),
                on="player_id", how="inner")
    p["sleeper_id"] = p["sid"]
    return p[["sleeper_id", "player_id", "week", "team", "opp", "home", "implied_total",
              "line_imputed", "own_status", "proj", "floor", "ceiling", "xfp_l3", "xfp_l8",
              "pts_l3", "n_games"]]


def ecr_points(week_proj: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Translate this week's FantasyPros ECR into points by giving each player
    the model projection of the model's same-ranked player at that position,
    then blend with the model using per-position MODEL_WEIGHT."""
    ecr = data.ecr_this_week()
    ecr = ecr[ecr["page_pos"].isin(["QB", "RB", "WR", "TE", "K", "DST"])].copy()
    ids = data.player_ids()[["fantasypros_id", "sleeper_id"]].dropna()
    ids["fantasypros_id"] = ids["fantasypros_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    ecr["fantasypros_id"] = ecr["fantasypros_id"].astype(str)
    ecr = ecr.merge(ids, on="fantasypros_id", how="left")
    # D/ST: sleeper id is the team code
    dst = ecr["page_pos"] == "DST"
    ecr.loc[dst, "sleeper_id"] = ecr.loc[dst, "team"].replace({"JAC": "JAX"})
    ecr = ecr.dropna(subset=["sleeper_id"])[["sleeper_id", "page_pos", "rank", "ecr"]]
    ecr["pos_rank"] = ecr.groupby("page_pos")["ecr"].rank(method="first").astype(int)

    w = week_proj.merge(ecr[["sleeper_id", "pos_rank", "ecr"]], on="sleeper_id", how="left")
    w = w.merge(sp[["sleeper_id", "position"]], on="sleeper_id", how="left")
    # half-PPR correction tested and rejected: made ECR slightly worse
    # (WR −0.3pp, significant) and left the 3-way blend unchanged.
    w["ecr_pts"] = ecr_to_points(w.dropna(subset=["position"]), ["position"], half_ppr_correct=False)
    w["blend"] = blend_scores(w)
    return w.drop(columns=["position"]), ecr


def blend_scores(w: pd.DataFrame) -> pd.Series:
    """Equal thirds of model, ECR-as-points and Sleeper's projection when all
    three exist — the best or tied-best combination at every position in the
    backtest (reports/backtest_summary.json -> three_way). Fallbacks: model+ECR
    with per-position MODEL_WEIGHT, then model+Sleeper average, then model."""
    mw = w["position"].map(MODEL_WEIGHT).fillna(0.5)
    has_e, has_s = w["ecr_pts"].notna(), w["sleeper_proj"].notna()
    return pd.Series(np.select(
        [has_e & has_s, has_e, has_s],
        [(w["proj"] + w["ecr_pts"] + w["sleeper_proj"]) / 3,
         mw * w["proj"] + (1 - mw) * w["ecr_pts"],
         (w["proj"] + w["sleeper_proj"]) / 2],
        default=w["proj"]), index=w.index)


def sleeper_live(season: int, weeks: list[int]) -> pd.DataFrame:
    fr = []
    for wk in weeks:
        try:
            fr.append(sleeper.historical_projections(season, wk, max_age_hours=3))
        except Exception as e:  # unofficial endpoint: degrade to model-only
            print(f"[advisor] Sleeper projections unavailable for week {wk}: {e}")
    if not fr:
        return pd.DataFrame(columns=["sleeper_id", "week", "sleeper_proj"])
    d = pd.concat(fr, ignore_index=True)
    return d.dropna(subset=["sleeper_proj"])[["sleeper_id", "week", "sleeper_proj"]]


# ------------------------------------------------------------------ main entry
def run(username: str, league_id: str | None = None, horizon: int = 4,
        n_waivers: int = 15) -> dict:
    st = sleeper.state()
    season, week = int(st["season"]), int(st["week"])
    lg, my = sleeper.resolve_league(username, season, league_id)
    slots = [s for s in lg["roster_positions"] if s not in ("BN", "IR", "TAXI")]
    unsupported = sorted({s for s in slots if s not in SLOT_ELIGIBLE})

    sp = sleeper.players()
    sp = sp[sp["position"].isin(MODELED | {"K", "DEF"})]
    mine_ids = set(my.get("players") or [])
    taken = sleeper.rostered_ids(lg["league_id"])
    fa_ids = set(sp.loc[sp["team"].notna() & (sp["active"] != False), "sleeper_id"]) - taken  # noqa: E712

    models = model.load()
    # project every player: the ECR->points ladder needs the whole league, not just my roster + FAs
    proj = project(sp[sp["team"].notna()], models, season, week)
    weeks = list(range(week, min(week + horizon, 19)))
    proj = proj.merge(sleeper_live(season, weeks), on=["sleeper_id", "week"], how="left")

    # --- this week: blend with ECR + Sleeper, zero out players ruled out
    wk = proj[proj["week"] == week]
    wk, ecr = ecr_points(wk, sp)
    me = sp[sp["sleeper_id"].isin(mine_ids)][["sleeper_id", "full_name", "position", "team", "injury_status"]]
    me = me.merge(wk, on="sleeper_id", how="left", suffixes=("", "_g"))
    # K / DEF: no model; score from ECR rank only (lower rank = better)
    kd = me["position"].isin(["K", "DEF"])
    me = me.merge(ecr[["sleeper_id", "pos_rank"]].rename(columns={"pos_rank": "kd_rank"}), on="sleeper_id", how="left")
    me.loc[kd, "blend"] = 100 - me.loc[kd, "kd_rank"].fillna(99)
    me["out"] = me["injury_status"].isin(OUT_THIS_WEEK)
    me["bye_or_played"] = me["proj"].isna() & ~kd
    tg = features.team_games()
    tg = tg[(tg["season"] == season) & (tg["week"] == week)].set_index("team")["completed"]
    nfl_team = me["team"].replace(SLEEPER_TO_NFLVERSE)
    me["no_play_reason"] = np.select(
        [~me["bye_or_played"], ~nfl_team.isin(tg.index), nfl_team.map(tg).fillna(False).astype(bool)],
        ["", "bye", "already played"], default="no NFL history to project from")
    me["score"] = me["blend"].where(~me["out"] & ~me["bye_or_played"], -1).fillna(-1)
    lineup, _ = optimal_lineup(me, slots, "score")
    starters = [sid for _, sid in lineup if sid]

    # --- waivers: change in optimal skill-position lineup points over horizon
    skill_slots = [s for s in slots if s not in ("K", "DEF")]
    ros = proj[proj["week"].isin(weeks)].copy()
    # no ECR for future weeks; model+Sleeper average had the best or near-best MAE
    ros["model_proj"] = ros["proj"]
    ros["proj"] = np.where(ros["sleeper_proj"].notna(), (ros["proj"] + ros["sleeper_proj"]) / 2, ros["proj"])
    ros = ros.merge(sp[["sleeper_id", "position", "injury_status", "full_name", "team"]], on="sleeper_id", how="left",
                    suffixes=("_nfl", ""))
    ros.loc[ros["injury_status"].isin(OUT_LONG_TERM), "proj"] = 0
    ros.loc[(ros["week"] == week) & ros["injury_status"].isin(OUT_THIS_WEEK), "proj"] = 0
    by_week = {w: g.set_index("sleeper_id") for w, g in ros.groupby("week")}

    def team_value(ids: set[str]) -> float:
        tot = 0.0
        for w in weeks:
            g = by_week.get(w)
            if g is None:
                continue
            pool = g.loc[g.index.intersection(list(ids))].reset_index()
            tot += optimal_lineup(pool, skill_slots, "proj")[1]
        return tot

    base = team_value(mine_ids)
    my_skill = [s for s in mine_ids if s in set(sp.loc[sp["position"].isin(MODELED), "sleeper_id"])]
    # candidate free agents: best horizon ppg per position
    fa = ros[ros["sleeper_id"].isin(fa_ids)].groupby("sleeper_id").agg(
        h_pts=("proj", "sum"), games=("proj", "size"), xfp_l3=("xfp_l3", "first"), xfp_l8=("xfp_l8", "first"))
    fa = fa.join(sp.set_index("sleeper_id")[["full_name", "position", "team", "injury_status"]])
    cand = fa.sort_values("h_pts", ascending=False).groupby("position").head(15)
    trending = sleeper.trending_adds()
    recs = []
    for sid, row in cand.iterrows():
        best = (-np.inf, None)
        for drop in my_skill:
            gain = team_value((mine_ids - {drop}) | {sid}) - base
            if gain > best[0]:
                best = (gain, drop)
        recs.append({
            "sleeper_id": sid, "player": row["full_name"], "pos": row["position"], "team": row["team"],
            "gain_pts": best[0], "drop": best[1], "horizon_pts": row["h_pts"], "games": row["games"],
            "usage_trend": (row["xfp_l3"] - row["xfp_l8"]) if pd.notna(row["xfp_l3"]) else np.nan,
            "trending_adds": trending.get(sid, 0), "injury": row["injury_status"],
        })
    allw = pd.DataFrame(recs).sort_values("gain_pts", ascending=False)
    # top overall, plus the best 3 at each position so one need doesn't hide the rest
    waivers = pd.concat([allw.head(n_waivers), allw.groupby("pos").head(3)]).drop_duplicates("sleeper_id")
    waivers = waivers.sort_values("gain_pts", ascending=False)
    names = sp.set_index("sleeper_id")["full_name"]
    waivers["drop"] = waivers["drop"].map(names)

    return {
        "season": season, "week": week, "league": lg, "slots": slots, "unsupported_slots": unsupported,
        "lineup": lineup, "roster": me, "starters": starters, "waivers": waivers,
        "horizon_weeks": weeks, "scoring": lg.get("scoring_settings", {}),
    }
