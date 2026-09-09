# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-09 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 5% of games and have a -100.0% ROI over the last 7 days. ROI is -26.38% over the last 30 days and 1.51% over the last 365.

| Date       | Away   | Home   | Away Pitcher    | Home Pitcher       |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:----------------|:-------------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-09 | MIN    | DET    | Zebby Matthews  | Keider Montero     |       43.69 |       56.31 | NA        | +158             | NA        | -105             |
| 2026-09-09 | TOR    | OAK    | Braydon Fisher  | Brady Basso        |       59.92 |       40.08 | -167      | -122             | 138       | n/a              |
| 2026-09-09 | STL    | SFG    | Andre Pallante  | Blade Tidwell      |       51.64 |       48.36 | -117      | +114             | -103      | +131             |
| 2026-09-09 | WSN    | SDP    | Jackson Kent    | Walker Buehler     |       39.6  |       60.4  | 153       | n/a              | -187      | -124             |
| 2026-09-09 | TEX    | SEA    | Cody Bradford   | Kade Anderson      |       48.19 |       51.81 | 127       | +132             | -154      | +114             |
| 2026-09-09 | CLE    | BAL    | Foster Griffin  | Shane Baz          |       47.37 |       52.63 | -118      | +136             | -102      | +110             |
| 2026-09-09 | NYM    | FLA    | Robert Stock    | Janson Junk        |       46.11 |       53.89 | 104       | +143             | -126      | +105             |
| 2026-09-09 | HOU    | PHI    | Hunter Brown    | Cristopher Sánchez |       41.13 |       58.87 | 129       | n/a              | -156      | -117             |
| 2026-09-09 | ANA    | BOS    | Ryan Johnson    | Jake Bennett       |       34.53 |       65.47 | 182       | n/a              | -223      | -153             |
| 2026-09-09 | COL    | NYY    | Tomoyuki Sugano | Will Warren        |       29.66 |       70.34 | 208       | n/a              | -258      | -188             |
| 2026-09-09 | TBD    | ATL    | Griffin Jax     |                    |       45.79 |       54.21 | 102       | +145             | -123      | +103             |
| 2026-09-09 | PIT    | CHW    | Lake Bachar     | Davis Martin       |       49.64 |       50.36 | 113       | +124             | -136      | +120             |
| 2026-09-09 | ARI    | KCR    | Zac Gallen      | Daniel Lynch IV    |       50.17 |       49.83 | -120      | +121             | 100       | +123             |
| 2026-09-09 | CHC    | MIL    | Kevin Gausman   | Logan Henderson    |       42.47 |       57.53 | 108       | n/a              | -131      | -111             |
| 2026-09-09 | CIN    | LAD    | Rhett Lowder    | Yoshinobu Yamamoto |       32.75 |       67.25 | 232       | n/a              | -291      | -165             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 20000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1562 |             -3 |               5 | 100.000%   | 99.980%        | 100.000%         | 61.955%    | 35.470%    | 23.430%  |
|  2 | LAD    |         1554 |              7 |               4 | 100.000%   | 100.000%       | 100.000%         | 58.510%    | 29.470%    | 18.485%  |
|  3 | CHC    |         1545 |             -1 |              -7 | 96.590%    | 0.020%         | 96.590%          | 22.095%    | 11.230%    | 6.555%   |
|  4 | NYY    |         1543 |              1 |              11 | 100.000%   | 20.455%        | 100.000%         | 35.530%    | 24.170%    | 11.335%  |
|  5 | BOS    |         1543 |              4 |              -6 | 99.915%    | 0.885%         | 99.915%          | 26.180%    | 17.620%    | 8.110%   |
|  6 | PHI    |         1536 |             -3 |              16 | 96.830%    | 10.405%        | 96.830%          | 19.850%    | 9.495%     | 5.180%   |
|  7 | ATL    |         1530 |             -2 |              -5 | 99.975%    | 89.595%        | 99.975%          | 23.645%    | 9.275%     | 5.140%   |
|  8 | TBD    |         1524 |              2 |              -3 | 100.000%   | 78.660%        | 100.000%         | 38.485%    | 23.230%    | 9.025%   |
|  9 | SDP    |         1522 |              2 |               6 | 51.395%    | <0.015%        | 51.395%          | 7.340%     | 2.715%     | 1.335%   |
| 10 | TOR    |         1512 |              4 |              15 | 43.380%    | <0.015%        | 43.380%          | 11.295%    | 4.640%     | 1.560%   |
| 11 | ARI    |         1512 |              5 |              -3 | 54.730%    | <0.015%        | 54.730%          | 6.560%     | 2.330%     | 1.105%   |
| 12 | PIT    |         1505 |              4 |               9 | 0.270%     | <0.015%        | 0.270%           | 0.035%     | 0.015%     | 0.005%   |
| 13 | CHW    |         1504 |             -3 |               1 | 88.925%    | 76.890%        | 88.925%          | 40.280%    | 14.465%    | 4.535%   |
| 14 | DET    |         1500 |             -1 |             -30 | 0.100%     | 0.020%         | 0.100%           | 0.025%     | 0.010%     | <0.015%  |
| 15 | NYM    |         1499 |              6 |               6 | <0.015%    | <0.015%        | <0.015%          | <0.015%    | <0.015%    | <0.015%  |
| 16 | HOU    |         1498 |              3 |              -5 | 82.625%    | 79.135%        | 82.625%          | 25.010%    | 8.395%     | 2.210%   |
| 17 | CLE    |         1498 |             -1 |               8 | 53.660%    | 22.895%        | 53.660%          | 16.490%    | 5.475%     | 1.535%   |
| 18 | FLA    |         1494 |             -8 |              -5 | 0.115%     | <0.015%        | 0.115%           | 0.005%     | <0.015%    | <0.015%  |
| 19 | STL    |         1492 |             -4 |               3 | 0.095%     | <0.015%        | 0.095%           | 0.005%     | <0.015%    | <0.015%  |
| 20 | BAL    |         1491 |             -9 |               1 | 1.460%     | <0.015%        | 1.460%           | 0.280%     | 0.085%     | 0.010%   |
| 21 | TEX    |         1486 |              6 |              -6 | 27.540%    | 20.060%        | 27.540%          | 6.030%     | 1.825%     | 0.435%   |
| 22 | KCR    |         1476 |             -2 |              17 | <0.015%    | <0.015%        | <0.015%          | <0.015%    | <0.015%    | <0.015%  |
| 23 | MIN    |         1475 |             -2 |              -7 | 1.435%     | 0.195%         | 1.435%           | 0.255%     | 0.055%     | 0.010%   |
| 24 | SEA    |         1474 |            -10 |             -14 | 0.960%     | 0.805%         | 0.960%           | 0.140%     | 0.030%     | <0.015%  |
| 25 | CIN    |         1473 |              2 |              -4 | <0.015%    | <0.015%        | <0.015%          | <0.015%    | <0.015%    | <0.015%  |
| 26 | WSN    |         1473 |             -6 |             -16 | <0.015%    | <0.015%        | <0.015%          | <0.015%    | <0.015%    | <0.015%  |
| 27 | SFG    |         1472 |              0 |              -1 | <0.015%    | <0.015%        | <0.015%          | <0.015%    | <0.015%    | <0.015%  |
| 28 | ANA    |         1452 |              5 |              12 | <0.015%    | <0.015%        | <0.015%          | <0.015%    | <0.015%    | <0.015%  |
| 29 | OAK    |         1431 |              7 |               0 | <0.015%    | <0.015%        | <0.015%          | <0.015%    | <0.015%    | <0.015%  |
| 30 | COL    |         1425 |             -1 |               0 | <0.015%    | <0.015%        | <0.015%          | <0.015%    | <0.015%    | <0.015%  |