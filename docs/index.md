# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-08 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 7% of games and have a -100.0% ROI over the last 7 days. ROI is -5.42% over the last 30 days and 0.32% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher   |   Away WinP |   Home WinP |   Away ML |   Away Threshold |   Home ML |   Home Threshold |
|:-----------|:-------|:-------|:---------------|:---------------|------------:|------------:|----------:|-----------------:|----------:|-----------------:|
| 2026-10-08 | CLE    | CHW    | Parker Messick | Hagen Smith    |       46.07 |       53.93 |      -115 |             +143 |      -105 |             +104 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when that outcome is still possible but did not occur in any simulation. A shown 0.00% means the team has no remaining path to that outcome.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1582 |              3 |              20 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 55.776%    | 38.864%  |
|  2 | LAD    |         1572 |              6 |              18 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 44.224%    | 29.080%  |
|  3 | TBD    |         1547 |              8 |              23 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 65.570%    | 23.804%  |
|  4 | NYY    |         1546 |             -7 |               3 | 100.000%   | 0.00%          | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
|  5 | SDP    |         1542 |             -4 |              20 | 100.000%   | 0.00%          | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
|  6 | CHC    |         1539 |              0 |              -6 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  7 | ATL    |         1527 |             -5 |              -3 | 100.000%   | 100.000%       | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
|  8 | BOS    |         1527 |              0 |             -16 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  9 | PHI    |         1521 |              0 |             -15 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 10 | ARI    |         1512 |              0 |               0 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 11 | CHW    |         1511 |              0 |               7 | 100.000%   | 0.00%          | 100.000%         | 74.790%    | 25.872%    | 6.226%   |
| 12 | CLE    |         1511 |              0 |              13 | 100.000%   | 100.000%       | 100.000%         | 25.210%    | 8.558%     | 2.026%   |
| 13 | PIT    |         1508 |              0 |               3 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 14 | DET    |         1507 |              0 |               7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 15 | BAL    |         1502 |              0 |              11 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 16 | TOR    |         1501 |              0 |             -11 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 17 | NYM    |         1501 |              0 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 18 | FLA    |         1497 |              0 |               3 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 19 | HOU    |         1491 |              0 |              -7 | 100.000%   | 100.000%       | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 20 | TEX    |         1484 |              0 |              -2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 21 | WSN    |         1481 |              0 |               8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 22 | STL    |         1480 |              0 |             -12 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 23 | SEA    |         1477 |              0 |               3 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 24 | MIN    |         1472 |              0 |              -3 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 25 | CIN    |         1464 |              0 |              -9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 26 | SFG    |         1463 |              0 |              -9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 27 | KCR    |         1463 |              0 |             -13 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 28 | ANA    |         1443 |              0 |              -9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 29 | OAK    |         1418 |              0 |             -13 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 30 | COL    |         1409 |              0 |             -16 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |