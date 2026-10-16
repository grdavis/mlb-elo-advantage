# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-16 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 50% of games and have a 72.75% ROI over the last 7 days. ROI is 13.41% over the last 30 days and 1.64% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher    |   Away WinP |   Home WinP |   Away ML |   Away Threshold |   Home ML |   Home Threshold |
|:-----------|:-------|:-------|:---------------|:----------------|------------:|------------:|----------:|-----------------:|----------:|-----------------:|
| 2026-10-16 | TBD    | CLE    | Nick Martinez  | Shawn Armstrong |       48.22 |       51.78 |      -112 |             +131 |      -108 |             +114 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when that outcome is still possible but did not occur in any simulation. A shown 0.00% means the team has no remaining path to that outcome.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1594 |             59 |              23 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 100.000%   | 75.708%  |
|  2 | LAD    |         1560 |             13 |               4 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 0.00%      | 0.00%    |
|  3 | NYY    |         1546 |             14 |              -1 | 100.000%   | 0.00%          | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
|  4 | SDP    |         1542 |             19 |              14 | 100.000%   | 0.00%          | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
|  5 | TBD    |         1542 |             47 |               8 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 37.026%    | 10.572%  |
|  6 | CHC    |         1539 |             13 |              -9 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  7 | ATL    |         1527 |             26 |              -1 | 100.000%   | 100.000%       | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
|  8 | BOS    |         1527 |              3 |             -10 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  9 | CLE    |         1522 |             11 |              22 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 62.974%    | 13.720%  |
| 10 | PHI    |         1521 |            -14 |             -12 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 11 | ARI    |         1512 |              8 |               7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 12 | PIT    |         1508 |             20 |               6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 13 | DET    |         1507 |              7 |              -5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 14 | CHW    |         1504 |             44 |               8 | 100.000%   | 0.00%          | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
| 15 | BAL    |         1502 |             14 |               4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 16 | TOR    |         1501 |            -39 |              -8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 17 | NYM    |         1501 |             -5 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 18 | FLA    |         1497 |             10 |               0 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 19 | HOU    |         1491 |            -13 |               1 | 100.000%   | 100.000%       | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 20 | TEX    |         1484 |            -24 |              -6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 21 | WSN    |         1481 |             29 |               7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 22 | STL    |         1480 |             -1 |              -9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 23 | SEA    |         1477 |            -44 |              -3 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 24 | MIN    |         1472 |              5 |               8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 25 | CIN    |         1464 |            -40 |              -1 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 26 | SFG    |         1463 |            -37 |             -11 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 27 | KCR    |         1463 |            -44 |             -15 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 28 | ANA    |         1443 |            -11 |              -9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 29 | OAK    |         1418 |            -73 |             -10 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 30 | COL    |         1409 |              0 |              -8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |