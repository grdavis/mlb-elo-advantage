# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-30 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 8% of games and have a 10.12% ROI over the last 7 days. ROI is -35.86% over the last 30 days and 1.47% over the last 365.

| Date       | Away   | Home   | Away Pitcher       | Home Pitcher   |   Away WinP |   Home WinP | Away ML   |   Away Threshold | Home ML   |   Home Threshold |
|:-----------|:-------|:-------|:-------------------|:---------------|------------:|------------:|:----------|-----------------:|:----------|-----------------:|
| 2026-09-30 | PHI    | ATL    | Cristopher Sánchez | Tyler Mahle    |       45.66 |       54.34 | NA        |             +146 | NA        |             +103 |
| 2026-09-30 | CHW    | HOU    | Sean Burke         | Hunter Brown   |       47.52 |       52.48 | 129       |             +135 | -155      |             +111 |
| 2026-09-30 | BOS    | NYY    | Sonny Gray         | Max Fried      |       44.28 |       55.72 | 112       |             +155 | -136      |             -103 |
| 2026-09-30 | CHC    | SDP    | Kevin Gausman      | Nick Pivetta   |       44.58 |       55.42 | 123       |             +153 | -149      |             -102 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1579 |              6 |              20 | 100.000%   | 100.000%       | 100.000%         | 64.018%    | 39.414%    | 27.460%  |
|  2 | LAD    |         1566 |              2 |              12 | 100.000%   | 100.000%       | 100.000%         | 64.812%    | 32.392%    | 21.348%  |
|  3 | NYY    |         1550 |              2 |              11 | 100.000%   | <0.006%        | 82.058%          | 43.026%    | 28.640%    | 12.470%  |
|  4 | SDP    |         1543 |              6 |              18 | 100.000%   | <0.006%        | 79.558%          | 28.790%    | 12.766%    | 6.920%   |
|  5 | CHC    |         1541 |             -4 |             -11 | 100.000%   | <0.006%        | 20.442%          | 7.192%     | 3.152%     | 1.750%   |
|  6 | TBD    |         1539 |              3 |              15 | 100.000%   | 100.000%       | 100.000%         | 48.768%    | 31.204%    | 12.576%  |
|  7 | ATL    |         1531 |             -1 |               0 | 100.000%   | 100.000%       | 79.742%          | 28.338%    | 10.120%    | 5.056%   |
|  8 | BOS    |         1530 |             -6 |             -16 | 100.000%   | <0.006%        | 17.942%          | 8.206%     | 4.960%     | 1.736%   |
|  9 | PHI    |         1523 |             -9 |             -14 | 100.000%   | <0.006%        | 20.258%          | 6.850%     | 2.156%     | 1.006%   |
| 10 | ARI    |         1512 |              2 |               3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 11 | CLE    |         1511 |              6 |              10 | 100.000%   | 100.000%       | 100.000%         | 53.812%    | 19.848%    | 5.652%   |
| 12 | CHW    |         1508 |              6 |               2 | 100.000%   | <0.006%        | 74.032%          | 35.310%    | 12.004%    | 3.218%   |
| 13 | PIT    |         1508 |              0 |               6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 14 | DET    |         1507 |              1 |               3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 15 | BAL    |         1502 |              3 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 16 | TOR    |         1501 |              0 |              -5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | NYM    |         1501 |             -2 |              10 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 18 | FLA    |         1497 |              0 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 19 | HOU    |         1494 |              5 |              -2 | 100.000%   | 100.000%       | 25.968%          | 10.878%    | 3.344%     | 0.808%   |
| 20 | TEX    |         1484 |             -2 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 21 | WSN    |         1481 |              0 |               1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | STL    |         1480 |             -5 |              -9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | SEA    |         1477 |              0 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 24 | MIN    |         1472 |              3 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | CIN    |         1464 |              1 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | SFG    |         1463 |             -4 |              -8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | KCR    |         1463 |             -7 |             -20 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1443 |              0 |              -8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1418 |             -8 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1409 |             -2 |             -13 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
