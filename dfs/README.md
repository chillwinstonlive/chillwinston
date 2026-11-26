# EPL DFS team builder: runbook

Goal: lock the max-expected-points FanTeam lineup each gameweek, publish it with a
hash, then score it against friends afterwards.

## One-off setup (PC)
```
git pull
pip install -r dfs/requirements.txt
```

## Gameweek routine (approx. 15 min)

| When | Step | Command / action |
|---|---|---|
| Fri eve / Sat am | 1. Fresh data | `python dfs/pipeline/fetch.py --gw 12` |
| | 2. FanTeam market | Contest lobby → download icon (upload template CSV) → save as `dfs/epl/GW12/players.csv` |
| | 3. Contest rules | Contest **Details → Rules**: formations, max per club, captain, stacking penalty, safety net → update `dfs/rules.md` |
| | 4. Team news | Claude checks press conferences / predicted XIs → `dfs/epl/GW12/overrides.csv` (`name,team,p_start,note`) |
| | 5. Project | `python dfs/pipeline/project.py --gw 12 --exclude NFO,CHE` (exclude teams not on the FanTeam slate) |
| | 6. Optimise | `python dfs/pipeline/optimise.py --gw 12 --budget 113 [--max-club 3] [--stacking]` |
| Before lock | 7. Lock | `python dfs/pipeline/lock.py --gw 12 --lock "2026-11-28 15:00" --tournament 1174816` |
| After GW | 8. Compare | "compare GW12" → friends' teams + FanTeam points screenshot |

## Data sources

| Source | What | Status |
|---|---|---|
| FPL API | players, xG/xA, minutes, injuries, set pieces, fixtures, per-match history | Working |
| ESPN API | per-player shots on target, saves, cards (every match); DraftKings 1X2 + totals (live and closing) | Working |
| FanTeam | prices, positions, IDs | Behind login → template CSV download |
| Understat | xG | Down ("we will come back soon") |
| FBref, Premier Injuries | stats, injuries | Block bots (403) |
| The Odds API | scorer/SoT props | Needs free key (not set) |

## Model (dfs/pipeline/project.py)
- Team goal means (λ) fitted to de-vigged DraftKings 1X2 + over/under (Poisson).
- Player xG/xA/SoT per 90: season + last 6 GWs, shrunk to positional prior (720 pseudo-mins).
- Rates rescaled by λ ÷ team's season xG per match (power 0.75).
- Minutes: weighted last-6 start rate × FPL availability; P(60), P(90) from recent starts.
- CS = P60 × e^(−λ_against); conceded = E[⌊G/2⌋]; GK saves = λ_against × (SoT per goal − 0.95).
- FanTeam position from the market CSV overrides FPL position (mis-classification edge).

## Backtest GW4–11 (closing odds, data before each GW only)

| Measure | Model | Recent form |
|---|---|---|
| Correlation with actual points | 0.47 | 0.41 |
| Avg actual pts of top-20 picks | 4.67 | 4.25 |
| Bias by position | within ±0.1 | - |

Top-end projections run ~10% hot (5+ projected → ~4.8 actual). Single-GW variance is
huge: judge over 10+ GWs.
