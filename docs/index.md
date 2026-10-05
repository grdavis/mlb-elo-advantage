# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-05 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 7% of games and have a -100.0% ROI over the last 7 days. ROI is -15.01% over the last 30 days and 1.93% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher   |   Away WinP |   Home WinP | Away ML   |   Away Threshold | Home ML   |   Home Threshold |
|:-----------|:-------|:-------|:---------------|:---------------|------------:|------------:|:----------|-----------------:|:----------|-----------------:|
| 2026-10-05 | CHW    | CLE    | Anthony Kay    | Gavin Williams |       45.86 |       54.14 | NA        |             +145 | NA        |             +103 |
| 2026-10-05 | NYY    | TBD    | Cam Schlittler | Freddy Peralta |       49.51 |       50.49 | -125      |             +125 | 104       |             +120 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when that outcome is still possible but did not occur in any simulation. A shown 0.00% means the team has no remaining path to that outcome.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1582 |             47 |              19 | 100.000%   | 100.000%       | 100.000%         | 91.966%    | 58.072%    | 41.008%  |
|  2 | LAD    |         1566 |             19 |              15 | 100.000%   | 100.000%       | 100.000%         | 57.364%    | 25.190%    | 16.416%  |
|  3 | NYY    |         1551 |             19 |               7 | 100.000%   | 0.00%          | 100.000%         | 33.952%    | 22.366%    | 9.198%   |
|  4 | SDP    |         1542 |             19 |              23 | 100.000%   | 0.00%          | 100.000%         | 8.034%     | 3.504%     | 1.932%   |
|  5 | TBD    |         1541 |             46 |              19 | 100.000%   | 100.000%       | 100.000%         | 66.048%    | 41.542%    | 15.356%  |
|  6 | CHC    |         1539 |             13 |             -12 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  7 | ATL    |         1533 |             32 |               0 | 100.000%   | 100.000%       | 100.000%         | 42.636%    | 13.234%    | 6.638%   |
|  8 | BOS    |         1527 |              3 |             -17 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  9 | PHI    |         1521 |            -14 |             -17 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 10 | CHW    |         1514 |             54 |              10 | 100.000%   | 0.00%          | 100.000%         | 70.790%    | 26.020%    | 6.988%   |
| 11 | ARI    |         1512 |              8 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 12 | PIT    |         1508 |             20 |               8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 13 | CLE    |         1507 |             -4 |              12 | 100.000%   | 100.000%       | 100.000%         | 29.210%    | 10.072%    | 2.464%   |
| 14 | DET    |         1507 |              7 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 15 | BAL    |         1502 |             14 |               8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 16 | TOR    |         1501 |            -39 |             -14 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 17 | NYM    |         1501 |             -5 |               9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 18 | FLA    |         1497 |             10 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 19 | HOU    |         1491 |            -13 |              -7 | 100.000%   | 100.000%       | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 20 | TEX    |         1484 |            -24 |               4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 21 | WSN    |         1481 |             29 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 22 | STL    |         1480 |             -1 |             -13 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 23 | SEA    |         1477 |            -44 |               1 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 24 | MIN    |         1472 |              5 |              -6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 25 | CIN    |         1464 |            -40 |              -8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 26 | SFG    |         1463 |            -37 |              -8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 27 | KCR    |         1463 |            -44 |             -13 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 28 | ANA    |         1443 |            -11 |              -7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 29 | OAK    |         1418 |            -73 |             -14 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 30 | COL    |         1409 |              0 |             -19 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |