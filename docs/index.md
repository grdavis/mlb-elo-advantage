# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-03 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 11% of games and have a 4.45% ROI over the last 7 days. ROI is -22.84% over the last 30 days and 1.93% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher      |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   |   Home Threshold |
|:-----------|:-------|:-------|:---------------|:------------------|------------:|------------:|:----------|:-----------------|:----------|-----------------:|
| 2026-10-03 | CHW    | CLE    | Hagen Smith    | Parker Messick    |       46.21 |       53.79 | NA        | +143             | NA        |             +105 |
| 2026-10-03 | ATL    | LAD    | Dylan Dodd     | Tarik Skubal      |       32.45 |       67.55 | 173       | n/a              | -212      |             -167 |
| 2026-10-03 | NYY    | TBD    | Gerrit Cole    | Drew Rasmussen    |       44.77 |       55.23 | 112       | +151             | -135      |             -101 |
| 2026-10-03 | SDP    | MIL    | Robbie Ray     | Jacob Misiorowski |       35.88 |       64.12 | 178       | n/a              | -219      |             -145 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when that outcome is still possible but did not occur in any simulation. A shown 0.00% means the team has no remaining path to that outcome.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1579 |              1 |              15 | 100.000%   | 100.000%       | 100.000%         | 63.166%    | 38.734%    | 26.482%  |
|  2 | LAD    |         1566 |              1 |              18 | 100.000%   | 100.000%       | 100.000%         | 63.432%    | 31.672%    | 20.374%  |
|  3 | NYY    |         1553 |              7 |              11 | 100.000%   | 0.00%          | 100.000%         | 53.972%    | 36.444%    | 16.484%  |
|  4 | SDP    |         1546 |              9 |              26 | 100.000%   | 0.00%          | 100.000%         | 36.834%    | 16.694%    | 8.950%   |
|  5 | TBD    |         1539 |             -3 |              20 | 100.000%   | 100.000%       | 100.000%         | 46.028%    | 28.912%    | 11.616%  |
|  6 | CHC    |         1539 |             -3 |              -8 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  7 | ATL    |         1532 |              1 |               0 | 100.000%   | 100.000%       | 100.000%         | 36.568%    | 12.900%    | 6.438%   |
|  8 | BOS    |         1527 |            -10 |             -13 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  9 | PHI    |         1521 |             -1 |             -18 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 10 | ARI    |         1512 |             -3 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 11 | CHW    |         1511 |              7 |               7 | 100.000%   | 0.00%          | 100.000%         | 48.380%    | 16.378%    | 4.576%   |
| 12 | CLE    |         1511 |             -1 |              15 | 100.000%   | 100.000%       | 100.000%         | 51.620%    | 18.266%    | 5.080%   |
| 13 | PIT    |         1508 |              2 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 14 | DET    |         1507 |             -2 |               6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 15 | BAL    |         1502 |              0 |               4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 16 | TOR    |         1501 |              2 |             -10 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 17 | NYM    |         1501 |             -2 |               8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 18 | FLA    |         1497 |              2 |              -2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 19 | HOU    |         1491 |             -2 |              -7 | 100.000%   | 100.000%       | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 20 | TEX    |         1484 |             -2 |               0 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 21 | WSN    |         1481 |              2 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 22 | STL    |         1480 |             -1 |             -15 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 23 | SEA    |         1477 |              2 |              -4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 24 | MIN    |         1472 |              2 |              -5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 25 | CIN    |         1464 |             -3 |              -7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 26 | SFG    |         1463 |             -2 |              -7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 27 | KCR    |         1463 |              2 |             -18 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 28 | ANA    |         1443 |             -2 |              -4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 29 | OAK    |         1418 |             -4 |              -9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 30 | COL    |         1409 |             -1 |             -17 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |