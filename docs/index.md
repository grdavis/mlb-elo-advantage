# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-21 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 7% of games and have a 33.71% ROI over the last 7 days. ROI is -30.05% over the last 30 days and 1.72% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher   |   Away WinP |   Home WinP |   Away ML | Away Threshold   |   Home ML |   Home Threshold |
|:-----------|:-------|:-------|:---------------|:---------------|------------:|------------:|----------:|:-----------------|----------:|-----------------:|
| 2026-09-21 | TOR    | BAL    | Trey Yesavage  | Shane Baz      |       50.24 |       49.76 |      -109 | +121             |      -110 |             +123 |
| 2026-09-21 | WSN    | DET    | DJ Herz        | River Ryan     |       42.12 |       57.88 |       113 | n/a              |      -136 |             -112 |
| 2026-09-21 | MIN    | SFG    | Zebby Matthews | Blade Tidwell  |       47.87 |       52.13 |      -121 | +133             |       101 |             +112 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1573 |              5 |              10 | 100.000%   | 100.000%       | 100.000%         | 63.930%    | 37.384%    | 25.896%  |
|  2 | LAD    |         1564 |              6 |              11 | 100.000%   | 100.000%       | 100.000%         | 62.634%    | 31.282%    | 20.474%  |
|  3 | CHC    |         1551 |              2 |               9 | 99.930%    | <0.006%        | 99.930%          | 21.902%    | 10.828%    | 6.398%   |
|  4 | NYY    |         1546 |              0 |               9 | 100.000%   | 0.826%         | 100.000%         | 30.600%    | 21.288%    | 9.346%   |
|  5 | TBD    |         1538 |              6 |              22 | 100.000%   | 99.174%        | 100.000%         | 49.474%    | 32.988%    | 13.350%  |
|  6 | SDP    |         1537 |              6 |              11 | 99.086%    | <0.006%        | 99.086%          | 15.978%    | 6.966%     | 3.804%   |
|  7 | BOS    |         1536 |             -6 |             -15 | 100.000%   | <0.006%        | 100.000%         | 20.140%    | 13.306%    | 5.370%   |
|  8 | ATL    |         1535 |              7 |               9 | 100.000%   | 100.000%       | 100.000%         | 20.556%    | 7.880%     | 4.220%   |
|  9 | PHI    |         1532 |             -2 |              -1 | 98.758%    | <0.006%        | 98.758%          | 14.770%    | 5.610%     | 2.796%   |
| 10 | PIT    |         1509 |              3 |               9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 11 | ARI    |         1507 |             -2 |              -3 | 2.226%     | <0.006%        | 2.226%           | 0.230%     | 0.050%     | 0.014%   |
| 12 | DET    |         1507 |             -3 |             -10 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 13 | TOR    |         1506 |             -5 |              -1 | 4.936%     | <0.006%        | 4.936%           | 1.154%     | 0.390%     | 0.128%   |
| 14 | CLE    |         1505 |              8 |               9 | 97.630%    | 55.886%        | 97.630%          | 40.428%    | 13.924%    | 3.736%   |
| 15 | CHW    |         1501 |              1 |              -5 | 95.938%    | 44.114%        | 95.938%          | 34.996%    | 11.394%    | 2.986%   |
| 16 | NYM    |         1497 |             -5 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | BAL    |         1494 |              2 |               0 | 0.038%     | <0.006%        | 0.038%           | 0.014%     | 0.008%     | <0.006%  |
| 18 | TEX    |         1491 |              7 |              12 | 68.286%    | 67.698%        | 68.286%          | 16.176%    | 4.688%     | 1.090%   |
| 19 | FLA    |         1491 |             -2 |             -10 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 20 | HOU    |         1486 |             -5 |              -4 | 33.084%    | 32.214%        | 33.084%          | 7.010%     | 2.010%     | 0.392%   |
| 21 | STL    |         1483 |            -12 |             -15 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | SEA    |         1480 |              2 |              -8 | 0.088%     | 0.088%         | 0.088%           | 0.008%     | 0.004%     | <0.006%  |
| 23 | WSN    |         1480 |              6 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 24 | KCR    |         1471 |             -6 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | SFG    |         1468 |              0 |               4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | MIN    |         1468 |              3 |              -8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | CIN    |         1460 |             -4 |             -12 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1448 |             -5 |              -9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1421 |             -9 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1414 |             -1 |             -11 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |