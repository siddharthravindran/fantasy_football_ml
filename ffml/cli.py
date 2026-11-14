"""Command line entry point.

    python -m ffml.cli train                   # refresh data + retrain (weekly, ~1 min)
    python -m ffml.cli backtest                # walk-forward eval vs FantasyPros ECR
    python -m ffml.cli week --user <sleeper>   # start/sit + waiver report
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from . import features, model, scoring

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def cmd_train(_):
    df, _ = features.build_dataset(include_future=False)
    m = model.train(df)
    p = model.save(m)
    print(f"trained on {df['pts'].notna().sum():,} player-games ({m['n_rounds']} rounds) -> {p}")


def cmd_backtest(_):
    from . import backtest
    import json
    df, _ = features.build_dataset(include_future=False)
    res = backtest.run(df)
    REPORTS.mkdir(exist_ok=True)
    res["weekly"].to_csv(REPORTS / "backtest_weekly.csv", index=False)
    (REPORTS / "backtest_summary.json").write_text(json.dumps(res["summary"], indent=2, default=float))
    print(json.dumps(res["summary"], indent=2, default=float))


def _win_prob(gap: float, pos_a: str, pos_b: str) -> float:
    """Backtest win rate of the higher-projected player at this gap
    (reports/confidence_by_gap.json); cross-position pairs average the two."""
    import json
    import bisect
    c = json.loads((REPORTS / "confidence_by_gap.json").read_text())
    b = min(bisect.bisect_right(c["bins"], abs(gap)) - 1, len(c["bins"]) - 2)
    rates = [c["win_rate"][p][b] for p in {pos_a, pos_b} if p in c["win_rate"]]
    return sum(rates) / len(rates) if rates else float("nan")


def _fmt(x, nd=1):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def cmd_week(args):
    from . import advisor
    mp = model.MODEL_DIR / "current.pkl"
    if not mp.exists() or time.time() - mp.stat().st_mtime > 6 * 86400:
        print("model missing or >6 days old — retraining first...")
        cmd_train(args)

    r = advisor.run(args.user, args.league, horizon=args.horizon, n_waivers=args.waivers)
    lg, me = r["league"], r["roster"].set_index("sleeper_id")
    me["injury_status"] = me["injury_status"].fillna("")
    out = [f"# Week {r['week']} — {lg['name']}\n"]

    diff = scoring.differs_from_half_ppr(r["scoring"])
    if diff:
        out.append("> ⚠️ League scoring differs from the half-PPR the model was trained on: "
                   + ", ".join(f"`{k}` {a}→{b}" for k, (a, b) in diff.items()) + "\n")
    if r["unsupported_slots"]:
        out.append(f"> ⚠️ Slots not handled (left empty in the optimizer): {r['unsupported_slots']}\n")

    hurt = [slot for slot, sid in r["lineup"] if sid is None or me.loc[sid, "score"] < 0]
    if hurt:
        out.append(f"> 🚨 No healthy, playing option for: {hurt}. Check the waiver targets below.\n")
    out.append("## Start\n\n| Slot | Player | Pos | Opp | Model | Floor–Ceiling | ECR pts | Sleeper | Score | Status |\n|---|---|---|---|---|---|---|---|---|---|")
    for slot, sid in r["lineup"]:
        if sid is None:
            out.append(f"| {slot} | — (no eligible player) | | | | | | | | |"); continue
        p = me.loc[sid]
        opp = "" if pd.isna(p.get("opp")) else ("vs " if p.get("home") == 1 else "@ ") + str(p["opp"])
        rng = "" if pd.isna(p.get("floor")) else f"{p['floor']:.1f}–{p['ceiling']:.1f}"
        out.append(f"| {slot} | {p['full_name']} | {p['position']} | {opp} | {_fmt(p.get('proj'))} | {rng} | "
                   f"{_fmt(p.get('ecr_pts'))} | {_fmt(p.get('sleeper_proj'))} | {_fmt(p['score'])} | {p['injury_status']} |")

    bench = me.drop(index=[s for s in r["starters"] if s in me.index]).sort_values("score", ascending=False)
    out.append("\n## Bench\n\n| Player | Pos | Model | Floor–Ceiling | ECR pts | Sleeper | Score | Note |\n|---|---|---|---|---|---|---|---|")
    for sid, p in bench.iterrows():
        note = "out" if p["out"] else p["no_play_reason"]
        rng = "" if pd.isna(p.get("floor")) else f"{p['floor']:.1f}–{p['ceiling']:.1f}"
        out.append(f"| {p['full_name']} | {p['position']} | {_fmt(p.get('proj'))} | {rng} | {_fmt(p.get('ecr_pts'))} | "
                   f"{_fmt(p.get('sleeper_proj'))} | {_fmt(p['score'])} | {note} {p['injury_status']} |")

    # close calls: bench player within 1.5 pts of a starter they could replace
    close = []
    from .advisor import SLOT_ELIGIBLE
    slot_of = {sid: slot for slot, sid in r["lineup"] if sid}
    st = me.loc[[s for s in r["starters"] if s in me.index]]
    for sid, b in bench[~bench["out"] & bench["proj"].notna()].iterrows():
        for ssid, s in st.iterrows():
            if b["position"] not in SLOT_ELIGIBLE.get(slot_of[ssid], set()):
                continue
            if 0 <= s["score"] - b["score"] < 1.5:
                p = _win_prob(s["score"] - b["score"], s["position"], b["position"])
                close.append(f"- **{s['full_name']}** ({_fmt(s['score'])}) over **{b['full_name']}** ({_fmt(b['score'])}) "
                             f"— wins ~{p:.0%} of the time at this gap; {b['full_name']}'s ceiling is "
                             f"{_fmt(b['ceiling'])} vs {_fmt(s['ceiling'])}")
    if close:
        out.append("\n## Close calls\n\n*Win % = how often the higher-projected player actually outscored the other "
                   "at this projection gap in the 2022–26 backtest.*\n\n" + "\n".join(close))

    w = r["waivers"]
    hw = r["horizon_weeks"]
    if w["gain_pts"].max() < 2:
        out.append(f"\n> ✅ **No free agent adds more than {max(w['gain_pts'].max(), 0):.1f} projected points to your "
                   f"lineup over weeks {hw[0]}–{hw[-1]}.** Your bench already covers it; keep your waiver "
                   "priority/FAAB. The list below is depth/handcuff interest only.")
    out.append(f"\n## Waiver targets (weeks {hw[0]}–{hw[-1]})\n\n`Gain` = change in your optimal lineup's projected points "
               "over the horizon if you add the player and drop the suggested player.\n\n"
               "| Player | Pos | Team | Gain | Drop | Proj pts (games) | Usage trend (xFP L3−L8) | Sleeper adds 48h |\n|---|---|---|---|---|---|---|---|")
    for _, x in w.iterrows():
        inj = f" ({x['injury']})" if isinstance(x["injury"], str) and x["injury"] else ""
        out.append(f"| {x['player']}{inj} | {x['pos']} | {x['team']} | "
                   f"{x['gain_pts']:+.1f} | {x['drop']} | {x['horizon_pts']:.1f} ({x['games']}) | {_fmt(x['usage_trend'])} | {x['trending_adds']:,} |")

    out.append("\n---\n*Score = equal-thirds blend of model, FantasyPros ECR (translated to points) and Sleeper's projection — "
               "best or tied-best at every position in backtesting. ECR comes from FantasyPros' full-PPR pages (the only "
               "ones archived); half-PPR ECR can differ for high-target players; model only where ECR is missing. K/DEF ranked by ECR alone. "
               "Lines for weeks without posted odds are estimated.*")
    text = "\n".join(out)
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"week_{r['season']}_{r['week']:02d}.md"
    path.write_text(text)
    print(text)
    print(f"\nsaved -> {path}")


def main():
    ap = argparse.ArgumentParser(prog="ffml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train").set_defaults(fn=cmd_train)
    sub.add_parser("backtest").set_defaults(fn=cmd_backtest)
    w = sub.add_parser("week")
    w.add_argument("--user", required=True, help="Sleeper username")
    w.add_argument("--league", help="Sleeper league id (needed if you're in several leagues)")
    w.add_argument("--horizon", type=int, default=4, help="weeks ahead for waiver value")
    w.add_argument("--waivers", type=int, default=15, help="waiver targets to show")
    w.set_defaults(fn=cmd_week)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
