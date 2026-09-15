# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-15 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 2% of games and have a -100.0% ROI over the last 7 days. ROI is -42.91% over the last 30 days and -0.71% over the last 365.

| Date       | Away   | Home   | Away Pitcher       | Home Pitcher   |   Away WinP |   Home WinP |   Away ML | Away Threshold   |   Home ML | Home Threshold   |
|:-----------|:-------|:-------|:-------------------|:---------------|------------:|------------:|----------:|:-----------------|----------:|:-----------------|
| 2026-09-15 | LAD    | CIN    | Yoshinobu Yamamoto | Rhett Lowder   |       64.88 |       35.12 |      -246 | -149             |       199 | n/a              |
| 2026-09-15 | CHW    | CLE    | Chris Murphy       | Foster Griffin |       48.71 |       51.29 |       128 | +129             |      -155 | +116             |
| 2026-09-15 | MIL    | PIT    | Jacob Misiorowski  | Lake Bachar    |       56.57 |       43.43 |      -217 | -106             |       177 | +160             |
| 2026-09-15 | OAK    | TBD    | Jack Perkins       | Griffin Jax    |       30.78 |       69.22 |       193 | n/a              |      -239 | -179             |
| 2026-09-15 | PHI    | WSN    | Cristopher Sánchez | Jackson Kent   |       57.03 |       42.97 |      -217 | -108             |       177 | +163             |
| 2026-09-15 | DET    | TOR    | Drew Anderson      | Braydon Fisher |       46.19 |       53.81 |       118 | +143             |      -142 | +105             |
| 2026-09-15 | BAL    | NYM    | Shane Baz          | Sean Manaea    |       47.16 |       52.84 |       113 | +137             |      -136 | +109             |
| 2026-09-15 | ATL    | CHC    | Martín Pérez       | Kevin Gausman  |       45.93 |       54.07 |       119 | +144             |      -143 | +104             |
| 2026-09-15 | NYY    | MIN    | Max Fried          | Bailey Ober    |       60.07 |       39.93 |      -180 | -123             |       148 | n/a              |
| 2026-09-15 | SFG    | STL    | Blade Tidwell      | Andre Pallante |       40.94 |       59.06 |       134 | n/a              |      -162 | -118             |
| 2026-09-15 | BOS    | TEX    | Patrick Sandoval   | Jacob deGrom   |       52.26 |       47.74 |       108 | +112             |      -130 | +134             |
| 2026-09-15 | KCR    | HOU    | Michael Wacha      | Hunter Brown   |       45.81 |       54.19 |       139 | +145             |      -168 | +103             |
| 2026-09-15 | SDP    | COL    | Walker Buehler     | Kyle Freeland  |       63.98 |       36.02 |      -194 | -144             |       159 | n/a              |
| 2026-09-15 | SEA    | ANA    | Logan Gilbert      | Ryan Johnson   |       49.73 |       50.27 |      -174 | +124             |       144 | +121             |
| 2026-09-15 | FLA    | ARI    | Janson Junk        | Michael Soroka |       43.58 |       56.42 |       125 | +159             |      -151 | -106             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 28571 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1568 |              6 |              12 | 100.000%   | 100.000%       | 100.000%         | 63.284%    | 36.586%    | 24.360%  |
|  2 | LAD    |         1558 |              4 |              11 | 100.000%   | 100.000%       | 100.000%         | 60.474%    | 30.377%    | 19.215%  |
|  3 | CHC    |         1549 |              4 |               4 | 98.537%    | <0.011%        | 98.537%          | 23.272%    | 11.736%    | 7.039%   |
|  4 | NYY    |         1546 |              3 |              15 | 100.000%   | 14.445%        | 100.000%         | 34.192%    | 24.336%    | 11.162%  |
|  5 | BOS    |         1542 |             -1 |              -2 | 99.996%    | 0.011%         | 99.996%          | 23.254%    | 16.079%    | 6.965%   |
|  6 | PHI    |         1534 |             -2 |              11 | 94.942%    | 2.741%         | 94.942%          | 16.608%    | 6.958%     | 3.658%   |
|  7 | TBD    |         1532 |              8 |               9 | 100.000%   | 85.545%        | 100.000%         | 42.638%    | 27.843%    | 11.585%  |
|  8 | SDP    |         1531 |              9 |               7 | 88.086%    | <0.011%        | 88.086%          | 14.207%    | 6.069%     | 3.143%   |
|  9 | ATL    |         1528 |             -2 |              -6 | 100.000%   | 97.259%        | 100.000%         | 20.136%    | 7.627%     | 3.882%   |
| 10 | TOR    |         1511 |             -1 |              10 | 38.067%    | <0.011%        | 38.067%          | 10.266%    | 3.920%     | 1.211%   |
| 11 | DET    |         1510 |             10 |             -14 | 1.369%     | 0.298%         | 1.369%           | 0.462%     | 0.172%     | 0.056%   |
| 12 | ARI    |         1509 |             -3 |              -6 | 18.361%    | <0.011%        | 18.361%          | 2.013%     | 0.644%     | 0.294%   |
| 13 | PIT    |         1506 |              1 |               5 | 0.067%     | <0.011%        | 0.067%           | 0.007%     | 0.004%     | <0.011%  |
| 14 | NYM    |         1502 |              3 |               8 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 15 | CHW    |         1500 |             -4 |             -10 | 89.367%    | 75.146%        | 89.367%          | 40.499%    | 13.059%    | 3.675%   |
| 16 | CLE    |         1497 |             -1 |               9 | 58.489%    | 24.553%        | 58.489%          | 18.949%    | 5.989%     | 1.600%   |
| 17 | STL    |         1495 |              3 |              -6 | 0.007%     | <0.011%        | 0.007%           | <0.011%    | <0.011%    | <0.011%  |
| 18 | FLA    |         1493 |             -1 |             -10 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 19 | BAL    |         1492 |              1 |              -7 | 2.531%     | <0.011%        | 2.531%           | 0.592%     | 0.165%     | 0.049%   |
| 20 | HOU    |         1491 |             -7 |              -8 | 75.542%    | 71.573%        | 75.542%          | 21.088%    | 6.213%     | 1.544%   |
| 21 | TEX    |         1484 |             -2 |               4 | 33.916%    | 27.787%        | 33.916%          | 7.893%     | 2.177%     | 0.550%   |
| 22 | SEA    |         1478 |              4 |             -11 | 0.665%     | 0.641%         | 0.665%           | 0.161%     | 0.049%     | 0.014%   |
| 23 | KCR    |         1477 |              1 |              17 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 24 | WSN    |         1474 |              1 |             -10 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 25 | SFG    |         1468 |             -4 |              -2 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 26 | MIN    |         1465 |            -10 |              -7 | 0.060%     | 0.004%         | 0.060%           | 0.007%     | <0.011%    | <0.011%  |
| 27 | CIN    |         1464 |             -9 |              -9 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 28 | ANA    |         1453 |              1 |               7 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 29 | OAK    |         1430 |             -1 |               0 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 30 | COL    |         1415 |            -10 |             -19 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |