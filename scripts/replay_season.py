"""Replay a Sleeper team's season: actual lineups vs what the tool would have
started, using only pre-week information.

    python -m scripts.replay_season --user <sleeper_username> [--weeks 1-9]

Projections come from the walk-forward backtest (model trained only on prior
seasons), blended exactly like the live tool with FantasyPros ECR (as points)
and Sleeper's archived weekly projections. Points are Sleeper's own league
scoring. K/DEF are held at the manager's actual choices.

Known bias (reported in the output): players with no stat line that week
(byes, inactives) have no projection, so the tool can't start them — on a
surprise inactive that's slight hindsight help the manager didn't have.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ffml import advisor, backtest, data, sleeper
from ffml.ecr import ecr_to_points

ROOT = Path(__file__).resolve().parents[1]
SKILL_SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"]


def week_projections(season: int, weeks: list[int]) -> pd.DataFrame:
    P = pd.read_parquet(ROOT / "reports/backtest_preds.parquet")
    P = P[(P["season"] == season) & P["week"].isin(weeks)]
    ids = pd.read_parquet(ROOT / "data/processed/dataset.parquet",
                          columns=["player_id", "season", "week", "sleeper_id", "fantasypros_id"])
    P = P.merge(ids.drop_duplicates(["player_id", "season", "week"]), on=["player_id", "season", "week"], how="left")
    for c in ["sleeper_id", "fantasypros_id"]:
        P[c] = P[c].astype("string").str.replace(r"\.0$", "", regex=True)
    # ECR -> points over the full slate (as live)
    ecr = backtest.weekly_ecr()
    P = P.merge(ecr.drop(columns=["pos"]), on=["season", "week", "fantasypros_id"], how="left")
    grp = ["season", "week", "position"]
    P["pos_rank"] = P.groupby(grp)["ecr"].rank(method="first")
    P["ecr_pts"] = ecr_to_points(P, grp, half_ppr_correct=False)
    sp = pd.concat([sleeper.historical_projections(season, w) for w in weeks])[["sleeper_id", "week", "sleeper_proj"]]
    P = P.merge(sp, on=["sleeper_id", "week"], how="left")
    has_e, has_s = P["ecr_pts"].notna(), P["sleeper_proj"].notna()
    P["blend"] = np.select([has_e & has_s, has_e, has_s],
                           [(P["proj"] + P["ecr_pts"] + P["sleeper_proj"]) / 3,
                            (P["proj"] + P["ecr_pts"]) / 2, (P["proj"] + P["sleeper_proj"]) / 2],
                           default=P["proj"])
    return P[["sleeper_id", "week", "position", "proj", "sleeper_proj", "blend"]]


def run(username: str, weeks: list[int]):
    st = sleeper.state()
    season = int(st["season"])
    lg, my = sleeper.resolve_league(username, season)
    sp = sleeper.players().set_index("sleeper_id")
    proj = week_projections(season, weeks)

    rows, missing_starters = [], []
    for w in weeks:
        ms = sleeper._get(f"/league/{lg['league_id']}/matchups/{w}")
        me = next(x for x in ms if x["roster_id"] == my["roster_id"])
        opp = next(x for x in ms if x["matchup_id"] == me["matchup_id"] and x["roster_id"] != me["roster_id"])
        others = [x["points"] for x in ms if x["roster_id"] != me["roster_id"]]
        pts = {k: float(v) for k, v in me["players_points"].items()}
        kdef = sum(me["starters_points"][8:10])
        actual_skill = sum(me["starters_points"][:8])

        pool = pd.DataFrame({"sleeper_id": me["players"]})
        pool["position"] = pool["sleeper_id"].map(sp["position"])
        pool = pool[pool["position"].isin(["QB", "RB", "WR", "TE"])]
        pool["actual"] = pool["sleeper_id"].map(pts).fillna(0.0)
        pool = pool.merge(proj[proj["week"] == w].drop(columns=["position"]), on="sleeper_id", how="left")
        for sid in me["starters"][:8]:
            if sid in set(pool["sleeper_id"]) and pool.loc[pool["sleeper_id"] == sid, "proj"].isna().all():
                missing_starters.append((w, sp["full_name"].get(sid, sid), pts.get(sid, 0.0)))

        def lineup_pts(score_col):
            avail = pool.dropna(subset=[score_col]) if score_col != "actual" else pool
            lu, _ = advisor.optimal_lineup(avail, SKILL_SLOTS, score_col)
            chosen = [sid for _, sid in lu if sid]
            return float(pool.set_index("sleeper_id").loc[chosen, "actual"].sum()), chosen

        res = {"week": w, "opp_pts": opp["points"], "others": others, "kdef": kdef, "actual": actual_skill}
        for name, col in [("tool", "blend"), ("model", "proj"), ("sleeper", "sleeper_proj"), ("hindsight", "actual")]:
            res[name], chosen = lineup_pts(col)
            if name == "tool":
                act = set(me["starters"][:8])
                res["tool_in"] = [sp["full_name"].get(s, s) for s in chosen if s not in act]
                res["tool_out"] = [sp["full_name"].get(s, s) for s in act if s not in set(chosen)]
        rows.append(res)

    R = pd.DataFrame(rows)

    def record(col):
        hw = mw = 0
        for _, r in R.iterrows():
            total = r[col] + r["kdef"]
            hw += total > r["opp_pts"]
            allp = sorted(r["others"] + [total])
            med = (allp[4] + allp[5]) / 2
            mw += total > med
        return hw, mw

    return R, record, missing_starters, lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--weeks", default="1-9")
    a = ap.parse_args()
    lo, hi = map(int, a.weeks.split("-"))
    R, record, missing, lg = run(a.user, list(range(lo, hi + 1)))
    pd.set_option("display.width", 200)
    print(R[["week", "actual", "tool", "model", "sleeper", "hindsight"]].round(1).to_string(index=False))
    tot = R[["actual", "tool", "model", "sleeper", "hindsight"]].sum()
    print("\nTotals (skill slots):", tot.round(1).to_dict())
    print("Efficiency vs hindsight:", (tot / tot["hindsight"]).round(3).to_dict())
    for c in ["actual", "tool", "model", "sleeper", "hindsight"]:
        h, m = record(c)
        print(f"  record with '{c}' lineups: H2H {h}-{len(R)-h}, median {m}-{len(R)-m}, total {h+m}-{2*len(R)-h-m}")
    print("\nWhere the tool would have differed:")
    for _, r in R.iterrows():
        if r["tool_in"]:
            print(f"  wk{r['week']}: start {r['tool_in']} instead of {r['tool_out']}  ({r['tool']-r['actual']:+.1f} pts)")
    if missing:
        print("\nYour starters with no stat line that week (tool couldn't have started them):", missing)


if __name__ == "__main__":
    main()
