# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-07 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 7% of games and have a -100.0% ROI over the last 7 days. ROI is -10.47% over the last 30 days and 0.32% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher   |   Away WinP |   Home WinP |   Away ML |   Away Threshold |   Home ML |   Home Threshold |
|:-----------|:-------|:-------|:---------------|:---------------|------------:|------------:|----------:|-----------------:|----------:|-----------------:|
| 2026-10-07 | CLE    | CHW    | Daniel Espino  | Sean Newcomb   |       43.21 |       56.79 |       108 |             +162 |      -130 |             -107 |
| 2026-10-07 | LAD    | ATL    | Tyler Glasnow  | Tyler Mahle    |       54.35 |       45.65 |      -156 |             +103 |       129 |             +146 |
| 2026-10-07 | TBD    | NYY    | Nick Martinez  | Max Fried      |       43.33 |       56.67 |       142 |             +161 |      -172 |             -107 |
| 2026-10-07 | MIL    | SDP    |                | Walker Buehler |       54.67 |       45.33 |      -107 |             +101 |      -112 |             +148 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when that outcome is still possible but did not occur in any simulation. A shown 0.00% means the team has no remaining path to that outcome.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1580 |              1 |              19 | 100.000%   | 100.000%       | 100.000%         | 81.370%    | 47.860%    | 33.174%  |
|  2 | LAD    |         1568 |              2 |              15 | 100.000%   | 100.000%       | 100.000%         | 81.846%    | 38.498%    | 25.278%  |
|  3 | NYY    |         1548 |             -5 |               6 | 100.000%   | 0.00%          | 100.000%         | 14.076%    | 9.202%     | 3.588%   |
|  4 | SDP    |         1545 |             -1 |              24 | 100.000%   | 0.00%          | 100.000%         | 18.630%    | 7.680%     | 4.188%   |
|  5 | TBD    |         1544 |              5 |              24 | 100.000%   | 100.000%       | 100.000%         | 85.924%    | 54.206%    | 20.782%  |
|  6 | CHC    |         1539 |              0 |              -7 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  7 | ATL    |         1530 |              1 |              -3 | 100.000%   | 100.000%       | 100.000%         | 18.154%    | 5.962%     | 2.860%   |
|  8 | BOS    |         1527 |              0 |             -20 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  9 | PHI    |         1521 |             -4 |             -17 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 10 | CHW    |         1517 |              6 |              10 | 100.000%   | 0.00%          | 100.000%         | 90.360%    | 33.346%    | 9.346%   |
| 11 | ARI    |         1512 |              0 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 12 | PIT    |         1508 |              0 |               7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 13 | DET    |         1507 |              0 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 14 | CLE    |         1505 |             -6 |              10 | 100.000%   | 100.000%       | 100.000%         | 9.640%     | 3.246%     | 0.784%   |
| 15 | BAL    |         1502 |              0 |               8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 16 | TOR    |         1501 |              0 |              -9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 17 | NYM    |         1501 |              0 |               4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 18 | FLA    |         1497 |              0 |               1 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 19 | HOU    |         1491 |              0 |              -5 | 100.000%   | 100.000%       | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 20 | TEX    |         1484 |              0 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 21 | WSN    |         1481 |              0 |               7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 22 | STL    |         1480 |              0 |             -13 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 23 | SEA    |         1477 |              0 |               0 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 24 | MIN    |         1472 |              0 |              -1 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 25 | CIN    |         1464 |              0 |             -10 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 26 | SFG    |         1463 |              0 |              -8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 27 | KCR    |         1463 |              0 |             -15 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 28 | ANA    |         1443 |              0 |              -5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 29 | OAK    |         1418 |              0 |             -14 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 30 | COL    |         1409 |              0 |             -17 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |