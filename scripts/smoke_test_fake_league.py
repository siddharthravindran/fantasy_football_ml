"""End-to-end smoke test of `ffml.cli week` against a simulated league
(real Sleeper players/ECR/projections; fake roster + league ownership)."""
import sys
from types import SimpleNamespace

import pandas as pd

from ffml import advisor, cli, sleeper

sp = sleeper.players()
sp = sp[sp["team"].notna() & sp["search_rank"].notna()].sort_values("search_rank")
pick = lambda pos, start, n: list(sp[sp.position == pos]["sleeper_id"].iloc[start:start + n])
mine = pick("QB", 6, 1) + pick("QB", 20, 1) + pick("RB", 10, 2) + pick("RB", 28, 3) + \
       pick("WR", 12, 2) + pick("WR", 30, 3) + pick("TE", 8, 2) + pick("K", 5, 1) + pick("DEF", 5, 1)
# league rosters ~ the 180 most-searched players
taken = set(sp["sleeper_id"].iloc[:180]) | set(mine)

LEAGUE = {"league_id": "fake", "name": "Smoke Test League",
          "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"] + ["BN"] * 6,
          "scoring_settings": {"rec": 0.5, "pass_td": 4}}
sleeper.resolve_league = lambda *a, **k: (LEAGUE, {"players": mine})
sleeper.rostered_ids = lambda _id: taken

cli.cmd_week(SimpleNamespace(user="fake", league=None, horizon=4, waivers=12))
