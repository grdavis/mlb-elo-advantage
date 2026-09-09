# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-08 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 6% of games and have a -100.0% ROI over the last 7 days. ROI is -26.38% over the last 30 days and 1.51% over the last 365.

| Date       | Away   | Home   | Away Pitcher    | Home Pitcher      |   Away WinP |   Home WinP |   Away ML | Away Threshold   |   Home ML | Home Threshold   |
|:-----------|:-------|:-------|:----------------|:------------------|------------:|------------:|----------:|:-----------------|----------:|:-----------------|
| 2026-09-08 | CLE    | BAL    | Tanner Bibee    | Brandon Young     |       47.26 |       52.74 |       102 | +137             |      -123 | +109             |
| 2026-09-08 | MIN    | DET    | Dean Kremer     | Drew Anderson     |       41.19 |       58.81 |       114 | n/a              |      -137 | -116             |
| 2026-09-08 | NYM    | FLA    | Sean Manaea     | Sandy Alcantara   |       44.9  |       55.1  |       103 | +151             |      -124 | -100             |
| 2026-09-08 | HOU    | PHI    | Hayden Wesneski | Andrew Painter    |       41.49 |       58.51 |       119 | n/a              |      -143 | -115             |
| 2026-09-08 | ANA    | BOS    | Reid Detmers    | Patrick Sandoval  |       36.31 |       63.69 |       123 | n/a              |      -149 | -142             |
| 2026-09-08 | COL    | NYY    | Gabriel Hughes  | Cam Schlittler    |       27.88 |       72.12 |       267 | n/a              |      -340 | -204             |
| 2026-09-08 | TBD    | ATL    | Freddy Peralta  | AJ Smith-Shawver  |       46.89 |       53.11 |      -105 | +139             |      -114 | +108             |
| 2026-09-08 | PIT    | CHW    | Bubba Chandler  | Sean Burke        |       45.9  |       54.1  |       120 | +145             |      -145 | +104             |
| 2026-09-08 | ARI    | KCR    | Corbin Burnes   | Michael Wacha     |       52.65 |       47.35 |      -120 | +110             |      -101 | +136             |
| 2026-09-08 | CHC    | MIL    | David Peterson  | Jacob Misiorowski |       42.33 |       57.67 |       178 | n/a              |      -219 | -111             |
| 2026-09-08 | TOR    | OAK    | José Soriano    | Jack Perkins      |       60.25 |       39.75 |      -186 | -123             |       153 | n/a              |
| 2026-09-08 | WSN    | SDP    | Riley Cornelio  | Casey Mize        |       42.59 |       57.41 |       155 | n/a              |      -188 | -110             |
| 2026-09-08 | TEX    | SEA    | Cal Quantrill   | Bryce Miller      |       51.55 |       48.45 |       113 | +115             |      -136 | +130             |
| 2026-09-08 | STL    | SFG    | Quinn Mathews   | Landen Roupp      |       50.36 |       49.64 |      -110 | +120             |      -110 | +124             |
| 2026-09-08 | CIN    | LAD    | Nick Lodolo     | Tarik Skubal      |       33.09 |       66.91 |       259 | n/a              |      -329 | -163             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 18867 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1561 |             -1 |               3 | 100.000%   | 99.740%        | 100.000%         | 60.121%    | 34.022%    | 22.627%  |
|  2 | LAD    |         1553 |              3 |               4 | 100.000%   | 99.995%        | 100.000%         | 55.886%    | 28.510%    | 17.989%  |
|  3 | BOS    |         1547 |              4 |              -3 | 99.963%    | 3.636%         | 99.963%          | 29.999%    | 20.920%    | 9.837%   |
|  4 | CHC    |         1546 |             -3 |              -6 | 98.611%    | 0.260%         | 98.611%          | 23.046%    | 11.661%    | 6.896%   |
|  5 | NYY    |         1542 |              1 |              10 | 99.995%    | 22.319%        | 99.995%          | 35.231%    | 23.406%    | 10.479%  |
|  6 | PHI    |         1538 |             -2 |              19 | 98.855%    | 11.396%        | 98.855%          | 20.454%    | 9.795%     | 5.480%   |
|  7 | ATL    |         1533 |              5 |              -5 | 99.995%    | 88.604%        | 99.995%          | 27.667%    | 11.645%    | 6.276%   |
|  8 | SDP    |         1521 |             -2 |               7 | 49.324%    | 0.005%         | 49.324%          | 6.811%     | 2.576%     | 1.224%   |
|  9 | TBD    |         1520 |             -6 |              -5 | 100.000%   | 74.045%        | 100.000%         | 34.802%    | 20.342%    | 7.654%   |
| 10 | TOR    |         1510 |              7 |              14 | 41.999%    | <0.016%        | 41.999%          | 10.733%    | 4.187%     | 1.484%   |
| 11 | ARI    |         1510 |              4 |              -2 | 52.102%    | <0.016%        | 52.102%          | 5.910%     | 1.781%     | 0.742%   |
| 12 | CHW    |         1507 |             -2 |               4 | 94.753%    | 86.442%        | 94.753%          | 47.612%    | 17.724%    | 5.682%   |
| 13 | DET    |         1502 |              4 |             -28 | 0.360%     | 0.037%         | 0.360%           | 0.074%     | 0.037%     | 0.005%   |
| 14 | PIT    |         1501 |             -2 |               5 | 0.233%     | <0.016%        | 0.233%           | 0.027%     | <0.016%    | <0.016%  |
| 15 | NYM    |         1497 |              8 |               7 | <0.016%    | <0.016%        | <0.016%          | <0.016%    | <0.016%    | <0.016%  |
| 16 | HOU    |         1496 |              3 |              -5 | 79.509%    | 76.594%        | 79.509%          | 22.229%    | 7.198%     | 1.961%   |
| 17 | FLA    |         1496 |             -4 |              -3 | 0.456%     | <0.016%        | 0.456%           | 0.042%     | 0.005%     | <0.016%  |
| 18 | CLE    |         1495 |             -9 |               5 | 47.565%    | 13.388%        | 47.565%          | 12.493%    | 4.177%     | 1.155%   |
| 19 | BAL    |         1494 |             -7 |               2 | 4.696%     | <0.016%        | 4.696%           | 0.853%     | 0.270%     | 0.085%   |
| 20 | STL    |         1493 |              0 |               2 | 0.419%     | <0.016%        | 0.419%           | 0.037%     | 0.005%     | 0.005%   |
| 21 | TEX    |         1482 |             -3 |              -8 | 26.035%    | 20.130%        | 26.035%          | 5.014%     | 1.489%     | 0.366%   |
| 22 | KCR    |         1478 |             -2 |              18 | <0.016%    | <0.016%        | <0.016%          | <0.016%    | <0.016%    | <0.016%  |
| 23 | SEA    |         1477 |             -3 |             -11 | 3.768%     | 3.276%         | 3.768%           | 0.769%     | 0.207%     | 0.042%   |
| 24 | CIN    |         1474 |              6 |              -3 | 0.005%     | <0.016%        | 0.005%           | <0.016%    | <0.016%    | <0.016%  |
| 25 | WSN    |         1474 |             -9 |             -15 | <0.016%    | <0.016%        | <0.016%          | <0.016%    | <0.016%    | <0.016%  |
| 26 | MIN    |         1473 |             -8 |              -6 | 1.357%     | 0.133%         | 1.357%           | 0.191%     | 0.042%     | 0.011%   |
| 27 | SFG    |         1471 |              1 |              -5 | <0.016%    | <0.016%        | <0.016%          | <0.016%    | <0.016%    | <0.016%  |
| 28 | ANA    |         1448 |             -1 |               6 | <0.016%    | <0.016%        | <0.016%          | <0.016%    | <0.016%    | <0.016%  |
| 29 | OAK    |         1432 |             13 |              -1 | <0.016%    | <0.016%        | <0.016%          | <0.016%    | <0.016%    | <0.016%  |
| 30 | COL    |         1426 |              2 |              -2 | <0.016%    | <0.016%        | <0.016%          | <0.016%    | <0.016%    | <0.016%  |