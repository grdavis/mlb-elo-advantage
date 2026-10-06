# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-06 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 6% of games and have a -100.0% ROI over the last 7 days. ROI is -10.47% over the last 30 days and 0.78% over the last 365.

| Date       | Away   | Home   | Away Pitcher       | Home Pitcher   |   Away WinP |   Home WinP |   Away ML |   Away Threshold |   Home ML |   Home Threshold |
|:-----------|:-------|:-------|:-------------------|:---------------|------------:|------------:|----------:|-----------------:|----------:|-----------------:|
| 2026-10-06 | LAD    | ATL    | Yoshinobu Yamamoto | Chris Sale     |       51.8  |       48.2  |      -110 |             +114 |      -109 |             +131 |
| 2026-10-06 | MIL    | SDP    | Dustin May         | Nick Pivetta   |       50.91 |       49.09 |       119 |             +118 |      -143 |             +127 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when that outcome is still possible but did not occur in any simulation. A shown 0.00% means the team has no remaining path to that outcome.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1582 |              3 |              22 | 100.000%   | 100.000%       | 100.000%         | 91.744%    | 58.056%    | 40.690%  |
|  2 | LAD    |         1566 |              0 |              14 | 100.000%   | 100.000%       | 100.000%         | 57.250%    | 25.134%    | 16.220%  |
|  3 | NYY    |         1548 |             -2 |               6 | 100.000%   | 0.00%          | 100.000%         | 13.772%    | 8.776%     | 3.648%   |
|  4 | TBD    |         1544 |              5 |              24 | 100.000%   | 100.000%       | 100.000%         | 86.228%    | 54.418%    | 20.864%  |
|  5 | SDP    |         1542 |             -1 |              22 | 100.000%   | 0.00%          | 100.000%         | 8.256%     | 3.716%     | 1.938%   |
|  6 | CHC    |         1539 |             -2 |              -8 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  7 | ATL    |         1533 |              2 |              -2 | 100.000%   | 100.000%       | 100.000%         | 42.750%    | 13.094%    | 6.404%   |
|  8 | BOS    |         1527 |             -3 |             -19 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  9 | PHI    |         1521 |             -2 |             -16 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 10 | CHW    |         1517 |              9 |              10 | 100.000%   | 0.00%          | 100.000%         | 90.284%    | 33.608%    | 9.444%   |
| 11 | ARI    |         1512 |              0 |               4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 12 | PIT    |         1508 |              0 |               7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 13 | DET    |         1507 |              0 |               6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 14 | CLE    |         1505 |             -6 |               8 | 100.000%   | 100.000%       | 100.000%         | 9.716%     | 3.198%     | 0.792%   |
| 15 | BAL    |         1502 |              0 |              10 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 16 | TOR    |         1501 |              0 |             -11 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 17 | NYM    |         1501 |              0 |               7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 18 | FLA    |         1497 |              0 |              -2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 19 | HOU    |         1491 |             -3 |              -5 | 100.000%   | 100.000%       | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 20 | TEX    |         1484 |              0 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 21 | WSN    |         1481 |              0 |               6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 22 | STL    |         1480 |              0 |             -15 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 23 | SEA    |         1477 |              0 |               0 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 24 | MIN    |         1472 |              0 |              -2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 25 | CIN    |         1464 |              0 |             -11 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 26 | SFG    |         1463 |              0 |              -6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 27 | KCR    |         1463 |              0 |             -16 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 28 | ANA    |         1443 |              0 |              -6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 29 | OAK    |         1418 |              0 |             -12 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 30 | COL    |         1409 |              0 |             -17 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |