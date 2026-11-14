"""Walk-forward backtest.

For each test season S: train on seasons < S, predict every week of S. Features
are as-of the prior week, so this mirrors using the tool live each week.

Compared against:
  * naive baselines (last-3 average, EWM average, season-to-date average)
  * FantasyPros weekly Expert Consensus Rankings (ECR), the de-facto benchmark
    for start/sit advice, on the same player-weeks.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data, model
from .ecr import ecr_to_points

TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]
# fantasy-relevant pool per position for start/sit comparisons (by ECR rank)
RELEVANT = {"QB": 24, "RB": 40, "WR": 50, "TE": 20}


# ------------------------------------------------------------------ ECR
def weekly_ecr() -> pd.DataFrame:
    """Map FantasyPros weekly ECR scrapes onto (season, week, fantasypros_id)."""
    r = data.ecr_history()
    r = r[r["page_type"].isin(["weekly-qb", "weekly-rb", "weekly-wr", "weekly-te"])].copy()
    r["date"] = pd.to_datetime(r["scrape_date"])
    sched = data.schedules()
    sched = sched[sched["game_type"] == "REG"]
    wk_end = (sched.assign(gameday=pd.to_datetime(sched["gameday"]))
                   .groupby(["season", "week"], as_index=False)["gameday"].max()
                   .sort_values("gameday"))
    # a scrape belongs to the first week whose last game is on/after the scrape date
    r = pd.merge_asof(r.sort_values("date"), wk_end, left_on="date", right_on="gameday",
                      direction="forward")
    # and must be within a week of that week's games (drops offseason scrapes)
    r = r[(r["gameday"] - r["date"]).dt.days <= 6]
    r = r.sort_values("date").groupby(["season", "week", "id"]).tail(1)  # latest scrape per week
    r["fantasypros_id"] = r["id"].astype(str)
    return r[["season", "week", "fantasypros_id", "ecr", "pos"]]


# ------------------------------------------------------------------ metrics
def _spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def _pairwise_acc(score, actual):
    """Fraction of player pairs where the higher-scored player actually
    outscored the other (ties in actual points skipped) — i.e. how often a
    head-to-head start/sit call would have been right."""
    s, a = np.asarray(score), np.asarray(actual)
    i, j = np.triu_indices(len(s), 1)
    keep = a[i] != a[j]
    return float(((s[i] > s[j]) == (a[i] > a[j]))[keep].mean()) if keep.any() else np.nan


def ranking_metrics(df: pd.DataFrame, score_cols: dict) -> pd.DataFrame:
    rows = []
    for (season, week, pos), g in df.groupby(["season", "week", "position"]):
        g = g.nsmallest(RELEVANT[pos], "ecr")
        if len(g) < 8:
            continue
        rec = {"season": season, "week": week, "position": pos, "n": len(g)}
        for name, col in score_cols.items():
            rec[f"spearman_{name}"] = _spearman(g[col], g["pts"])
            rec[f"pairwise_{name}"] = _pairwise_acc(g[col].values, g["pts"].values)
        rows.append(rec)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ run
def paired_ci(rk: pd.DataFrame, a: str, b: str, metric: str = "pairwise") -> dict:
    """Mean per-position-week difference a−b (percentage points) with 95% CI."""
    out = {}
    for pos, g in rk.groupby("position"):
        d = (g[f"{metric}_{a}"] - g[f"{metric}_{b}"]).dropna() * 100
        se = d.std() / np.sqrt(len(d))
        out[pos] = {"diff_pp": round(d.mean(), 2), "ci_lo": round(d.mean() - 1.96 * se, 2),
                    "ci_hi": round(d.mean() + 1.96 * se, 2), "n_weeks": int(len(d))}
    return out


def sleeper_projections(seasons) -> pd.DataFrame:
    from . import sleeper
    import nflreadpy as nfl
    cur_s, cur_w = data.current_season(), nfl.get_current_week()
    fr = []
    for s in seasons:
        for w in range(1, 19):
            if s == cur_s and w > cur_w:
                break  # don't cache projections for weeks not yet played
            fr.append(sleeper.historical_projections(s, w))
    d = pd.concat(fr, ignore_index=True)
    d = d[d["sleeper_proj"].notna()]
    return d[["sleeper_id", "season", "week", "sleeper_proj", "last_modified"]]


def run(df: pd.DataFrame, seasons=TEST_SEASONS, per_position: bool = False,
        compare_sleeper: bool = True) -> dict:
    df = df[df["pts"].notna()].copy()
    preds = []
    for s in seasons:
        tr = df[(df["season"] < s) & (df["season"] >= 2013)]
        te = df[df["season"] == s]
        if te.empty:
            continue
        m = model.train_per_position(tr) if per_position else model.train(tr)
        p = model.predict(m, te)
        preds.append(p)
        rounds = ({k: v["n_rounds"] for k, v in m["per_position"].items()} if per_position else m["n_rounds"])
        print(f"[backtest] {s}: trained on {len(tr):,} rows (rounds {rounds}), tested on {len(te):,}")
    P = pd.concat(preds, ignore_index=True)

    # --- point accuracy vs naive baselines (players with >=1 prior game)
    P["season_avg"] = P["pts_std_season"].fillna(P["pts_l8"])
    acc = P[P["n_games"] > 0]
    point = {}
    for pos, g in acc.groupby("position"):
        point[pos] = {
            name: float((g[col] - g["pts"]).abs().mean())
            for name, col in [("model", "proj"), ("last3_avg", "pts_l3"),
                              ("ewm_avg", "pts_ewm"), ("season_avg", "season_avg")]
        }
    cover = acc[(acc["pts"] >= acc["floor"]) & (acc["pts"] <= acc["ceiling"])]
    interval_coverage = len(cover) / len(acc)

    # --- ranking vs FantasyPros ECR on identical player-weeks
    ecr = weekly_ecr()
    P["fantasypros_id"] = P["fantasypros_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    grp = ["season", "week", "position"]
    # ECR -> points over the whole slate, exactly as the live tool does
    W = P.merge(ecr.drop(columns=["pos"]), on=["season", "week", "fantasypros_id"], how="left")
    W["pos_rank"] = W.groupby(grp)["ecr"].rank(method="first")
    W["ecr_pts"] = ecr_to_points(W, grp, half_ppr_correct=False)
    W["ecr_half_pts"] = ecr_to_points(W, grp, half_ppr_correct=True)
    R = W[W["ecr"].notna()].copy()
    R["neg_ecr"] = -R["ecr"]
    # simple ensemble: average the model's and the experts' within-week ranks
    R["model_rank"] = R.groupby(grp)["proj"].rank(ascending=False)
    R["ecr_rank"] = R.groupby(grp)["ecr"].rank()
    R["ecr_half_rank"] = R.groupby(grp)["ecr_half_pts"].rank(ascending=False)
    R["neg_blend"] = -(R["model_rank"] + R["ecr_rank"]) / 2
    R["neg_blend_half"] = -(R["model_rank"] + R["ecr_half_rank"]) / 2
    rk = ranking_metrics(R, {"model": "proj", "ecr": "neg_ecr", "ecr_half": "ecr_half_pts",
                             "blend": "neg_blend", "blend_half": "neg_blend_half", "last3": "pts_l3"})

    summary = {
        "point_mae_by_position": point,
        "floor_ceiling_coverage": interval_coverage,
        "ranking_by_position": rk.groupby("position")[[c for c in rk.columns if c.startswith(("spearman", "pairwise"))]].mean().round(4).to_dict("index"),
        "ranking_by_season": rk.groupby("season")[[c for c in rk.columns if c.startswith("pairwise")]].mean().round(4).to_dict("index"),
        "n_ranked_player_weeks": int(len(R)),
        "weeks_compared": int(rk[["season", "week"]].drop_duplicates().shape[0]),
    }
    # paired weekly win rate: in how many position-weeks did the model beat ECR?
    rk["model_beats_ecr"] = rk["pairwise_model"] > rk["pairwise_ecr"]
    summary["model_beats_ecr_share_of_position_weeks"] = float(rk["model_beats_ecr"].mean())
    summary["ci_model_vs_ecr"] = paired_ci(rk, "model", "ecr")
    summary["ci_blend_vs_ecr"] = paired_ci(rk, "blend", "ecr")
    summary["ci_ecr_half_vs_ecr"] = paired_ci(rk, "ecr_half", "ecr")
    summary["ci_blend_half_vs_blend"] = paired_ci(rk, "blend_half", "blend")

    out = {"summary": summary, "preds": P, "ranked": R, "weekly": rk}
    if compare_sleeper:
        out.update(_sleeper_comparison(P, R, seasons, summary))
    return out


def _sleeper_comparison(P, R, seasons, summary) -> dict:
    """Three-way comparison on player-weeks that have model, ECR AND a Sleeper
    projection. Sleeper's archive shows some projections edited after games
    (notably 2022, 2025) — any leakage there favors Sleeper, so we also report
    the 'clean' seasons where edits stop ~1 day after the week."""
    sp = sleeper_projections(seasons)
    P = P.copy()
    P["sleeper_id"] = P["sleeper_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    R = R.copy()
    R["sleeper_id"] = R["sleeper_id"].astype("string").str.replace(r"\.0$", "", regex=True)

    # point accuracy (Sleeper projects points, so MAE is comparable)
    M = P[P["n_games"] > 0].merge(sp, on=["sleeper_id", "season", "week"], how="inner")
    summary["sleeper_point_mae_by_position"] = {
        pos: {"model": float((g["proj"] - g["pts"]).abs().mean()),
              "sleeper": float((g["sleeper_proj"] - g["pts"]).abs().mean()),
              "avg_model_sleeper": float(((g["proj"] + g["sleeper_proj"]) / 2 - g["pts"]).abs().mean()),
              "n": int(len(g))}
        for pos, g in M.groupby("position")}

    R3 = R.merge(sp, on=["sleeper_id", "season", "week"], how="inner")
    g = R3.groupby(["season", "week", "position"])
    R3["sleeper_rank"] = g["sleeper_proj"].rank(ascending=False)
    R3["neg_blend3"] = -(R3["model_rank"] + R3["ecr_rank"] + R3["sleeper_rank"]) / 3
    R3["neg_blend3_half"] = -(R3["model_rank"] + R3["ecr_half_rank"] + R3["sleeper_rank"]) / 3
    # what the live tool actually computes: equal-thirds average of points
    R3["blend3_pts"] = (R3["proj"] + R3["ecr_pts"] + R3["sleeper_proj"]) / 3
    R3["blend3_half_pts"] = (R3["proj"] + R3["ecr_half_pts"] + R3["sleeper_proj"]) / 3
    scorers = {"model": "proj", "ecr": "neg_ecr", "ecr_half": "ecr_half_pts", "sleeper": "sleeper_proj",
               "blend": "neg_blend", "blend3": "neg_blend3", "blend3_half": "neg_blend3_half",
               "blend3_pts": "blend3_pts", "blend3_half_pts": "blend3_half_pts"}
    rk3 = ranking_metrics(R3, scorers)
    clean = rk3["season"].isin([2023, 2024, 2026])
    pw = [f"pairwise_{k}" for k in scorers]
    summary["three_way"] = {
        "n_player_weeks": int(len(R3)),
        "pairwise_by_position": rk3.groupby("position")[pw].mean().round(4).to_dict("index"),
        "pairwise_by_season": rk3.groupby("season")[pw].mean().round(4).to_dict("index"),
        "pairwise_clean_seasons_by_position": rk3[clean].groupby("position")[pw].mean().round(4).to_dict("index"),
        "ci_model_vs_sleeper": paired_ci(rk3, "model", "sleeper"),
        "ci_ecr_vs_sleeper": paired_ci(rk3, "ecr", "sleeper"),
        "ci_blend_vs_sleeper": paired_ci(rk3, "blend", "sleeper"),
        "ci_blend3_vs_ecr": paired_ci(rk3, "blend3", "ecr"),
        "ci_blend3_half_pts_vs_blend3_pts": paired_ci(rk3, "blend3_half_pts", "blend3_pts"),
        "ci_blend3_half_pts_vs_ecr": paired_ci(rk3, "blend3_half_pts", "ecr"),
        "ci_blend3_half_pts_vs_sleeper": paired_ci(rk3, "blend3_half_pts", "sleeper"),
    }
    # point accuracy of the live blend (where all three exist and the player played)
    M3 = R3[R3["n_games"] > 0]
    summary["three_way"]["mae_by_position"] = {
        pos: {c: round(float((g[c] - g["pts"]).abs().mean()), 3)
              for c in ["proj", "sleeper_proj", "ecr_pts", "ecr_half_pts", "blend3_pts", "blend3_half_pts"]}
        for pos, g in M3.groupby("position")}
    return {"weekly_three_way": rk3}


# ------------------------------------------------------------------ confidence
GAP_BINS = [0, 0.5, 1, 2, 3, 5, 8, 99]


def confidence_by_gap(P: pd.DataFrame, score: str = "proj") -> dict:
    """How often the higher-projected of two same-position, fantasy-relevant
    players actually outscored the other, by projection gap. Turns a
    projection difference into a start/sit win probability."""
    rows = []
    for (_, _, pos), g in P.groupby(["season", "week", "position"]):
        g = g.nlargest(RELEVANT[pos], score)
        pr, a = g[score].to_numpy(), g["pts"].to_numpy()
        i, j = np.triu_indices(len(g), 1)
        keep = a[i] != a[j]
        rows.append(pd.DataFrame({"position": pos, "gap": np.abs(pr[i] - pr[j])[keep],
                                  "win": ((pr[i] > pr[j]) == (a[i] > a[j]))[keep]}))
    D = pd.concat(rows)
    D["bin"] = pd.cut(D["gap"], GAP_BINS, right=False, labels=False)
    t = D.groupby(["position", "bin"])["win"].mean().round(3)
    return {"bins": GAP_BINS,
            "win_rate": {pos: [float(t.get((pos, b), np.nan)) for b in range(len(GAP_BINS) - 1)]
                         for pos in RELEVANT}}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from . import features
    root = Path(__file__).resolve().parents[1]
    per_pos = "--per-position" in sys.argv
    tag = "_per_position" if per_pos else ""
    df, _ = features.build_dataset(include_future=False)
    res = run(df, per_position=per_pos)
    (root / "reports").mkdir(exist_ok=True)
    res["weekly"].to_csv(root / f"reports/backtest_weekly{tag}.csv", index=False)
    if "weekly_three_way" in res:
        res["weekly_three_way"].to_csv(root / f"reports/backtest_weekly_three_way{tag}.csv", index=False)
    res["preds"][["player_id", "player_display_name", "position", "season", "week", "team", "opp", "pts",
                  "proj", "floor", "ceiling", "pts_l3"]].to_parquet(root / f"reports/backtest_preds{tag}.parquet")
    if not per_pos:
        (root / "reports/confidence_by_gap.json").write_text(json.dumps(confidence_by_gap(res["preds"]), indent=2))
    with open(root / f"reports/backtest_summary{tag}.json", "w") as f:
        json.dump(res["summary"], f, indent=2, default=float)
    print(json.dumps(res["summary"], indent=2, default=float))
