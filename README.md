# Fantasy Football Start/Sit & Waiver Model

A LightGBM projection model for half-PPR fantasy football, benchmarked against FantasyPros expert consensus rankings (ECR) in a walk-forward backtest. It's connected to a Sleeper league to produce a weekly lineup and a waiver list ranked by **marginal value to your roster**.

```
python -m ffml.cli week --user <sleeper_username>
```

## Results (walk-forward, 2022 – 2026 wk 10)

Each test season is predicted by a model trained only on earlier seasons. Features are computed as of the prior week, so these numbers reflect live use.

**Point accuracy (MAE, half-PPR points; players with ≥1 prior game)**

| Pos | Model | Last-3 avg | EWM avg | Season avg |
|---|---|---|---|---|
| QB | **5.50** | 6.53 | 6.22 | 6.33 |
| RB | **4.11** | 4.43 | 4.21 | 4.26 |
| WR | **3.89** | 4.17 | 3.94 | 4.02 |
| TE | **3.03** | 3.34 | 3.13 | 3.19 |

**Start/sit accuracy vs. the experts.** For every pair of fantasy-relevant players at a position (top 24 QB / 40 RB / 50 WR / 20 TE by ECR), how often did the higher-ranked player actually score more? The comparison covers 22,194 player-weeks over 72 weeks, all on identical player-weeks.

| Pos | Model | FantasyPros ECR | Model + ECR blend | Last-3 avg |
|---|---|---|---|---|
| QB | 61.4% | 61.5% | **61.8%** | 57.3% |
| RB | 64.4% | **64.9%** | **64.9%** | 60.7% |
| WR | 60.7% | **62.4%** | 61.9% | 57.2% |
| TE | 59.5% | 59.2% | **59.9%** | 54.1% |

Paired difference vs. ECR per position-week, with a 95% CI:

| Pos | Model − ECR | Blend − ECR |
|---|---|---|
| QB | −0.1 pp [−1.0, +0.8] | +0.3 pp [−0.3, +0.9] |
| RB | −0.5 pp [−1.1, +0.1] | 0.0 pp [−0.4, +0.4] |
| WR | **−1.7 pp [−2.3, −1.0]** | **−0.5 pp [−0.9, −0.1]** |
| TE | +0.3 pp [−0.9, +1.5] | +0.7 pp [−0.0, +1.5] |

**What this says:**
- The model alone is statistically tied with the experts at QB, RB and TE, and beats every naive baseline at every position.
- **Experts are measurably better at WR.** My best guess is that they price in news the box score doesn't capture yet: depth-chart changes, route participation, coverage matchups.
- Blending model and ECR never does significantly worse than ECR except at WR. It's the best point estimate at QB and TE, though neither gain is significant.
- Floor/ceiling (15th/85th quantile models) contains **69.5%** of outcomes against a nominal 70%. The intervals are calibrated.

### vs. Sleeper's projections (and the three-source blend)

Sleeper's in-app projections (sourced from Rotowire) are archived on an unofficial endpoint. This comparison uses 20,942 player-weeks where all three sources exist.

| Pos | Model | FantasyPros ECR | Sleeper | Model + ECR + Sleeper |
|---|---|---|---|---|
| QB | 61.4% | 61.4% | 60.2% | **61.6%** |
| RB | 64.4% | 64.9% | 65.3% | **65.4%** |
| WR | 60.9% | **62.4%** | 62.2% | **62.4%** |
| TE | 59.6% | 59.2% | 59.8% | **59.9%** |

| Pos | Model − Sleeper (95% CI) | 3-way blend − ECR (95% CI) | MAE: model / Sleeper / average |
|---|---|---|---|
| QB | **+1.2 pp [+0.3, +2.1]** | +0.2 pp [−0.3, +0.8] | **5.89** / 6.24 / 5.85 |
| RB | **−1.0 pp [−1.6, −0.3]** | **+0.5 pp [+0.1, +0.9]** | 4.24 / **4.02** / 4.07 |
| WR | **−1.3 pp [−1.9, −0.6]** | −0.1 pp [−0.4, +0.3] | 4.05 / **3.90** / 3.93 |
| TE | −0.2 pp [−1.5, +1.1] | +0.7 pp [−0.0, +1.4] | 3.15 / 3.06 / **3.05** |

- **The live tool's equal-thirds *point* average** matches the rank-average version. It beats FantasyPros significantly at RB (+0.4 pp) and TE (+0.9 pp), and Sleeper significantly at QB (+1.5 pp). It's statistically tied with the best single source everywhere else.
- **No single source wins everywhere.** The model is significantly better than Sleeper at QB, and Sleeper is significantly better at RB and WR.
- **An equal-weight average of all three ranks is the best or tied-best at every position.** It beats FantasyPros significantly at RB. The live tool uses it: an equal-thirds point blend for this week, and model+Sleeper for future weeks, where no ECR exists.
- **Possible leakage check.** Sleeper's archive shows some projections edited weeks after the games (57% of 2022 rows, 23% of 2025, but ≤12% in 2023/2024/2026). Any leak would *help* Sleeper. Sleeper isn't unusually strong in 2022 or 2025, and the per-position pattern is the same on the clean seasons alone (`three_way.pairwise_clean_seasons_by_position`). So there's no sign it distorts the comparison.

### Ablation: starting-QB features

These features add the listed starting QB (from the schedule; 96.6% match with the actual passing leader), whether he's new, his recent production and EPA, and each player's usage *with that specific QB*. QB point error dropped 5.78 → **5.50**, mostly because the model now knows whether a QB is the listed starter. Start/sit rankings didn't change at any position (±0.1 pp), since QB changes touch only ~9% of player-weeks. The features stay because they capture real cases. Example: Fannin's target share is 21% with Watson vs 7% in his one 2026 Sanders start (24% across 2025 Sanders starts).

### Ablation: coordinator-tendency features (rejected)

Coordinator *identity* isn't in free data and would be sparse, so I tested what coordinators leave in the play-by-play instead.
- **Offense:** plays per game, neutral-situation pass rate, pass rate over expected (PROE), short-target share.
- **Defense:** yards and EPA allowed per target on short (<10 air yards) vs deep targets, short-target share faced, rush EPA allowed.
- **Interaction:** the player's average target depth matched to what this defense allows at that depth.

The features capture real signal. For example, CIN 2026 allows 6.9 yards per short target vs a league average of 5.5, and 8.2 per deep target vs 10.7. But they didn't improve predictions. Paired change vs. the current model:

| | QB | RB | WR | TE |
|---|---|---|---|---|
| Live-blend start/sit accuracy | −0.09 pp | −0.09 pp | 0.00 pp | −0.21 pp (all n.s.) |
| MAE | **+0.028** (worse) | +0.003 | −0.001 | **+0.010** (worse) |

Most of this information is already absorbed elsewhere. Player usage shares × Vegas implied totals cover volume and pace, and points allowed by position plus Vegas cover matchup. The extra columns mostly gave the trees more room to fit noise. The code stays behind `USE_TENDENCIES` in [ffml/features.py](ffml/features.py), and the results are in `reports/*_tendencies_ablation.*`.

### Ablation: one model vs. separate models per position

Separate per-position models (each with its own early-stopped round count) were slightly worse at every position. Pairwise accuracy was 0.05–0.5 pp lower and MAE 0.002–0.03 pts higher. Floor/ceiling calibration also degraded, from 69.7% to 66.5% coverage. Sharing data across positions helps more than specializing, especially for TE and QB (fewer rows). The live tool keeps the single model, with `position` as a categorical feature. Reproduce with `python -m ffml.backtest --per-position`.

**Ablation:** adding injury-report features (own status + usage left behind by teammates ruled Out or Doubtful) improved RB pairwise accuracy 64.0→64.4% and lowered MAE at every position. It moved WR very little (60.6→60.7%). See `reports/backtest_summary_v1_no_injury.json`.

## How it works

```
nflverse (nflreadpy)                       Sleeper API (public)
  weekly stats, snaps, expected FP,          league settings, rosters,
  schedules + Vegas lines, injury reports    free agents, injury status, trending adds
  FantasyPros ECR archive                              │
          │                                            │
     features.py ── as-of join ──► model.py (LightGBM: mean + q15 + q85)
          │                                            │
     backtest.py (walk-forward vs ECR)        advisor.py: blend w/ live ECR →
                                              lineup optimizer → waiver marginal value
```

- **Leakage-free by construction.** Rolling stats are computed *inclusive* of each game. Each prediction row then takes the latest history row strictly before its week (`merge_asof(..., allow_exact_matches=False)`). The same code builds backtest rows and future-week rows, so training and serving can't drift apart.
- **Features (~90):** rolling 3/8-game and EWM usage (targets, carries, target share, air-yards share, WOPR, snap %, expected fantasy points), production, volatility, Vegas implied team total / spread / total, rest, dome/wind, opponent points allowed to the position, official injury status, usage left behind by injured teammates, and draft capital/experience for players with little history.
- **ECR → points.** ECR is a ranking, but FLEX decisions need points. The tool maps each player's ECR position rank onto the model's projection ladder for the full slate (so ECR WR12 gets the model's WR12 projection). It then averages model, ECR-points and Sleeper's projection in equal thirds. If a source is missing, it falls back to the per-position model/ECR weights in [reports/blend_weights.md](reports/blend_weights.md), then to model+Sleeper, then to the model alone.
- **Lineup optimizer:** greedy fill from the most restrictive slot outward. This is exact for nested eligibility sets (QB ⊂ SUPER_FLEX, RB ⊂ FLEX…). Reads `roster_positions` from the league.
- **Waivers:** for each free-agent candidate × each possible drop, the tool recomputes your optimal lineup's projected points over the next *N* weeks (default 4). Free agents are ranked by the gain. This answers "does this player make *my* team better?", not "who is good?". Lines for weeks without posted odds are estimated from both teams' season-average lines.

- **Trade finder:** values every 1-for-1 swap, plus 2-for-1s that pair any player with one of your *surplus* players (those adding <3 lineup points). For both teams it measures the change in optimal-lineup points, split into regular season and fantasy playoffs. A deal is *realistic* when it helps you and costs the other team ≤1 point. Value is pure lineup math: depth-as-insurance, the other manager's own opinions, and roster cuts aren't modeled. Lineup scoring is a memoized pure-Python greedy, so a full scan of ~100 packages × 9 rosters takes seconds after the ~80 s projection build.

## Usage

```bash
python3.12 -m venv ~/venvs/fantasy-football
~/venvs/fantasy-football/bin/pip install -r requirements.txt

python -m ffml.cli train                       # refresh data + retrain (auto-runs if model >6 days old)
python -m ffml.cli backtest                    # reproduce the tables above (~1.5 min)
python -m ffml.cli week --user <name> [--league <id>] [--horizon 4]
python -m ffml.cli trades --user <name> [--give "Player A + Player B"] # trade finder (~1.5 min)
```

The weekly report prints to the terminal and is saved to `reports/week_<season>_<week>.md`. Run it after Friday's injury report for the best inputs: injury features and ECR both update through the week.

## Limitations

- Trained on half-PPR. If your league's scoring differs, the report warns you, but the projections don't re-weight.
- **ECR is full-PPR.** nflverse archives only FantasyPros' PPR RB/WR/TE pages, both historically and live. Half-PPR consensus can rank high-target players lower; for example, Week 10 2026 Fannin/Kincaid flip between the PPR and half-PPR pages. I tested a correction: map the PPR rank onto a PPR-points ladder, then subtract 0.5 × expected receptions ([ffml/ecr.py](ffml/ecr.py)). It made ECR slightly *worse* (WR −0.3 pp, significant; RB/TE −0.2 pp, n.s.) and left the live three-way blend unchanged (±0.1 pp). The PPR ordering costs nothing measurable, so the live tool doesn't apply the correction.
- Sleeper projections come from an undocumented endpoint. If it fails, the tool falls back to model+ECR.
- K and DEF are ranked by ECR only (not modeled). IDP slots aren't supported.
- **Live injury news:** for the current week, Sleeper's injury status (which updates after the official Friday report) overrides the official one, and the more severe wins. A teammate counts as vacating usage only if he played in the last 2 weeks. Long absences are already reflected in teammates' recent usage, so a player out since Week 5 being ruled out again changes nothing.
- Projections are conditional on the player playing. Out/IR players are zeroed using Sleeper status, and "Questionable" goes into the model as a feature.
- WR is the weak spot (see above). The obvious next features are route participation and depth-chart order (nflverse `load_participation`, `load_depth_charts`).
- No in-season retraining inside the backtest (retrained once per season). The live tool retrains weekly.
- Players already locked (games played this week) are treated as having no remaining game. The optimizer doesn't account for lineup locks.

## Data

All from [nflverse](https://github.com/nflverse) via `nflreadpy` (stats, snaps, ff_opportunity expected points, schedules/lines, injuries, FantasyPros ECR archive) and the public [Sleeper API](https://docs.sleeper.com). Raw data is cached under `data/` (gitignored).
