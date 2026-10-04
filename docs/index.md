# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-04 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 15% of games and have a 4.45% ROI over the last 7 days. ROI is -22.84% over the last 30 days and 1.93% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher    |   Away WinP |   Home WinP |   Away ML | Away Threshold   |   Home ML |   Home Threshold |
|:-----------|:-------|:-------|:---------------|:----------------|------------:|------------:|----------:|:-----------------|----------:|-----------------:|
| 2026-10-04 | SDP    | MIL    | Michael King   | Logan Henderson |       37.78 |       62.22 |       113 | n/a              |      -136 |             -134 |
| 2026-10-04 | ATL    | LAD    |                | Blake Snell     |       37.37 |       62.63 |       130 | n/a              |      -157 |             -136 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when that outcome is still possible but did not occur in any simulation. A shown 0.00% means the team has no remaining path to that outcome.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1580 |              1 |              15 | 100.000%   | 100.000%       | 100.000%         | 78.446%    | 46.464%    | 32.454%  |
|  2 | LAD    |         1569 |              3 |              19 | 100.000%   | 100.000%       | 100.000%         | 78.798%    | 37.624%    | 24.816%  |
|  3 | NYY    |         1551 |              5 |              10 | 100.000%   | 0.00%          | 100.000%         | 34.018%    | 22.356%    | 9.130%   |
|  4 | SDP    |         1544 |              4 |              23 | 100.000%   | 0.00%          | 100.000%         | 21.554%    | 9.002%     | 5.008%   |
|  5 | TBD    |         1541 |              2 |              21 | 100.000%   | 100.000%       | 100.000%         | 65.982%    | 41.608%    | 15.570%  |
|  6 | CHC    |         1539 |             -6 |             -11 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  7 | ATL    |         1530 |              1 |              -5 | 100.000%   | 100.000%       | 100.000%         | 21.202%    | 6.910%     | 3.420%   |
|  8 | BOS    |         1527 |             -7 |             -15 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  9 | PHI    |         1521 |             -4 |             -15 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 10 | CHW    |         1514 |              9 |               8 | 100.000%   | 0.00%          | 100.000%         | 70.744%    | 26.080%    | 7.176%   |
| 11 | ARI    |         1512 |              0 |               7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 12 | PIT    |         1508 |              0 |               4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 13 | CLE    |         1507 |             -4 |               8 | 100.000%   | 100.000%       | 100.000%         | 29.256%    | 9.956%     | 2.426%   |
| 14 | DET    |         1507 |              0 |               8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 15 | BAL    |         1502 |              0 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 16 | TOR    |         1501 |              0 |             -13 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 17 | NYM    |         1501 |              0 |               6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 18 | FLA    |         1497 |              0 |               1 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 19 | HOU    |         1491 |             -6 |              -8 | 100.000%   | 100.000%       | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 20 | TEX    |         1484 |              0 |               1 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 21 | WSN    |         1481 |              0 |               4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 22 | STL    |         1480 |              0 |             -16 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 23 | SEA    |         1477 |              0 |              -2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 24 | MIN    |         1472 |              0 |              -3 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 25 | CIN    |         1464 |              0 |              -5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 26 | SFG    |         1463 |              0 |              -5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 27 | KCR    |         1463 |              0 |             -14 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 28 | ANA    |         1443 |              0 |              -3 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 29 | OAK    |         1418 |              0 |             -11 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 30 | COL    |         1409 |              0 |             -16 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |