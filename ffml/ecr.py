"""Turn FantasyPros ECR (a ranking) into half-PPR points.

ECR for RB/WR/TE comes from FantasyPros' *full-PPR* pages (the only ones
nflverse archives). A PPR rank implies a PPR point value; subtracting half a
point per expected reception converts it to half-PPR:

    ppr_ladder   = sort(model_proj + 0.5 * exp_receptions)   # slate, per position-week
    ecr_ppr_pts  = ppr_ladder[ecr_rank]
    ecr_half_pts = ecr_ppr_pts - 0.5 * exp_receptions(player)

The uncorrected version maps the rank straight onto the model's half-PPR
ladder, which keeps the PPR ordering. QB rankings are format-neutral, so QBs
are never adjusted. Expected receptions = the player's pre-game receptions EWM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PPR_PAGES = {"RB", "WR", "TE"}


def ecr_to_points(w: pd.DataFrame, group_cols: list[str], rank_col: str = "pos_rank",
                  proj_col: str = "proj", rec_col: str = "receptions_ewm",
                  half_ppr_correct: bool = True) -> pd.Series:
    """`w` holds the whole slate (ranked and unranked players); `rank_col` is the
    1-based ECR rank within the group, NaN if unranked. Groups must be
    position-level (include 'position')."""
    out = pd.Series(np.nan, index=w.index, dtype=float)
    for _, g in w.groupby(group_cols):
        has = g[rank_col].notna()
        if not has.any():
            continue
        adj = half_ppr_correct and g["position"].iloc[0] in PPR_PAGES
        rec = g[rec_col].fillna(0).to_numpy() if adj else np.zeros(len(g))
        ladder = np.sort((g[proj_col].to_numpy() + 0.5 * rec))[::-1]
        idx = (g.loc[has, rank_col].astype(int).to_numpy() - 1).clip(0, len(ladder) - 1)
        pts = ladder[idx] - 0.5 * rec[has.to_numpy()]
        out.loc[g.index[has]] = pts
    return out
