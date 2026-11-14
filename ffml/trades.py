"""Trade finder: 1-for-1 swaps valued by *marginal lineup points*.

Same idea as the waiver ranking: a player's value to a team is how much that
team's optimal starting lineup gains (over the remaining weeks) by having him.
A trade is worth proposing when it helps you AND doesn't hurt the other team
(so they have a reason to accept). Regular season and fantasy playoffs are
reported separately, since a title run cares more about weeks 15-17.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import advisor, model, sleeper

OUT_LONG = advisor.OUT_LONG_TERM


def _league_projections(season: int, first_week: int, last_week: int) -> pd.DataFrame:
    sp = sleeper.players()
    sp = sp[sp["position"].isin(advisor.MODELED) & sp["team"].notna()]
    proj = advisor.project(sp, model.load(), season, first_week)
    weeks = list(range(first_week, last_week + 1))
    proj = proj[proj["week"].isin(weeks)].merge(
        advisor.sleeper_live(season, weeks), on=["sleeper_id", "week"], how="left")
    # same rest-of-season rule as waivers: average model and Sleeper where both exist
    proj["proj"] = np.where(proj["sleeper_proj"].notna(), (proj["proj"] + proj["sleeper_proj"]) / 2, proj["proj"])
    proj = proj.merge(sp[["sleeper_id", "position", "full_name", "injury_status"]], on="sleeper_id")
    proj.loc[proj["injury_status"].isin(OUT_LONG), "proj"] = 0
    return proj


def find_trades(username: str, give_sets: list[list[str]] | None = None, league_id: str | None = None,
                regular_end: int = 14, playoff_end: int = 17, min_their_gain: float = -1.0,
                replacement_depth: int = 1) -> dict:
    """`replacement_depth`: how many of the best free agents per position (per
    week) every team may plug in. Without this, a player who only covers a bye
    looks worth his whole projection, when the real alternative is a waiver
    pickup — value is measured *over replacement*. 0 disables it."""
    st = sleeper.state()
    season, week = int(st["season"]), int(st["week"])
    lg, my = sleeper.resolve_league(username, season, league_id)
    first = week + 1  # this week's lineups are effectively set
    slots = [s for s in lg["roster_positions"] if s in advisor.SLOT_ELIGIBLE and s not in ("K", "DEF")]
    proj = _league_projections(season, first, playoff_end)
    names = proj.drop_duplicates("sleeper_id").set_index("sleeper_id")["full_name"]
    pos_of = proj.drop_duplicates("sleeper_id").set_index("sleeper_id")["position"]

    users = {u["user_id"]: (u.get("metadata") or {}).get("team_name") or u["display_name"]
             for u in sleeper.league_users(lg["league_id"])}
    rosters = {r["roster_id"]: {"team": users.get(r["owner_id"], "?"), "ids": set(r.get("players") or [])}
               for r in sleeper.rosters(lg["league_id"])}
    rostered = set().union(*(r["ids"] for r in rosters.values()))

    # plain dict lookups: {week: {sleeper_id: (proj, position)}}
    by_week = {w: dict(zip(g["sleeper_id"], zip(g["proj"].astype(float), g["position"])))
               for w, g in proj.groupby("week")}
    # replacement level: pseudo-players = the k-th best free agent at each position, each week
    repl_ids, replacement = set(), {}
    fa = proj[~proj["sleeper_id"].isin(rostered)]
    first_rows = fa[fa["week"] == fa["week"].min()].set_index("sleeper_id")
    ok = advisor.active_role(first_rows, week)
    fa = fa[fa["sleeper_id"].isin(ok[ok].index)]  # healthy FAs with a current role only
    for (w, pos), g in fa.groupby(["week", "position"]):
        top = g.nlargest(replacement_depth, "proj")
        for k, (_, row) in enumerate(top.iterrows()):
            rid = f"REPL_{pos}_{k}"
            by_week[w][rid] = (float(row["proj"]), pos)
            repl_ids.add(rid)
            replacement.setdefault(pos, {})[w] = f"{row['full_name']} ({row['proj']:.1f})"
    reg_weeks = [w for w in range(first, regular_end + 1)]
    po_weeks = [w for w in range(regular_end + 1, playoff_end + 1)]
    ordered_slots = [advisor.SLOT_ELIGIBLE[s] for s in
                     sorted(slots, key=lambda s: len(advisor.SLOT_ELIGIBLE[s]))]
    cache: dict = {}

    def value(ids: set, weeks) -> float:
        """Sum of optimal-lineup points; same greedy as advisor.optimal_lineup
        (exact for nested slot eligibility), without pandas, memoized."""
        ids = set(ids) | repl_ids  # every team can always fill from the waiver wire
        key = (frozenset(ids), tuple(weeks))
        if key in cache:
            return cache[key]
        tot = 0.0
        for w in weeks:
            pw = by_week.get(w, {})
            pool = sorted((pw[s] for s in ids if s in pw), reverse=True)
            used = [False] * len(pool)
            for elig in ordered_slots:
                for i, (pts, pos) in enumerate(pool):
                    if not used[i] and pos in elig:
                        used[i] = True
                        tot += pts
                        break
        cache[key] = tot
        return tot

    me_id = my["roster_id"]
    mine = rosters[me_id]["ids"]

    # how much each of my players is worth to MY lineup (what I'd lose without him)
    surplus = []
    base_r, base_p = value(mine, reg_weeks), value(mine, po_weeks)
    for sid in mine:
        if sid in pos_of.index:
            surplus.append({"player": names[sid], "pos": pos_of[sid],
                            "loss_regular": base_r - value(mine - {sid}, reg_weeks),
                            "loss_playoffs": base_p - value(mine - {sid}, po_weeks)})
    surplus = pd.DataFrame(surplus).sort_values(["loss_playoffs", "loss_regular"])

    by_name = {names[s]: s for s in mine if s in names.index}
    if give_sets:
        missing = [n for gs in give_sets for n in gs if n not in by_name]
        if missing:
            raise SystemExit(f"Not on your roster (or not a QB/RB/WR/TE): {missing}")
        sets = [[by_name[n] for n in gs] for gs in give_sets]
    else:
        # default packages: every player alone, plus 2-for-1s pairing any player
        # with a "surplus" one (adds < 3 pts to my lineup over the whole horizon)
        modeled = [s for s in mine if s in pos_of.index]
        spare = [by_name[p] for p, lr, lp in surplus[["player", "loss_regular", "loss_playoffs"]].values
                 if lr + lp < 3]
        sets = [[s] for s in modeled]
        # skip (b, a) when (a, b) already exists, i.e. both are spare
        sets += [[a, b] for i, a in enumerate(modeled) for b in spare if b != a
                 and not (a in spare and modeled.index(b) < i)]
    offers = []
    for rid, r in rosters.items():
        if rid == me_id:
            continue
        theirs = r["ids"]
        t_r, t_p = value(theirs, reg_weeks), value(theirs, po_weeks)
        for gset in sets:
            for get_id in theirs:
                if get_id not in pos_of.index:
                    continue
                # (a 2-for-1 forces them to cut someone; their worst bench player rarely matters to lineup value)
                m_new, t_new = (mine - set(gset)) | {get_id}, (theirs - {get_id}) | set(gset)
                my_r, my_p = value(m_new, reg_weeks) - base_r, value(m_new, po_weeks) - base_p
                th_r, th_p = value(t_new, reg_weeks) - t_r, value(t_new, po_weeks) - t_p
                offers.append({"give": " + ".join(names[g] for g in gset), "get": names[get_id], "get_pos": pos_of[get_id],
                               "with": r["team"], "my_gain_reg": my_r, "my_gain_playoffs": my_p,
                               "their_gain_reg": th_r, "their_gain_playoffs": th_p})
    o = pd.DataFrame(offers)
    o["my_gain_total"] = o["my_gain_reg"] + o["my_gain_playoffs"]
    o["their_gain_total"] = o["their_gain_reg"] + o["their_gain_playoffs"]
    realistic = o[(o["my_gain_total"] > 0) & (o["their_gain_total"] >= min_their_gain)]
    return {"weeks": (reg_weeks, po_weeks), "surplus": surplus, "all": o, "league": lg, "replacement": replacement,
            "realistic": realistic.sort_values("my_gain_total", ascending=False)}
