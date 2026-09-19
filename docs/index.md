# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-19 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 4% of games and have a -39.0% ROI over the last 7 days. ROI is -33.97% over the last 30 days and -0.11% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher     |   Away WinP |   Home WinP |   Away ML | Away Threshold   |   Home ML | Home Threshold   |
|:-----------|:-------|:-------|:---------------|:-----------------|------------:|------------:|----------:|:-----------------|----------:|:-----------------|
| 2026-09-19 | DET    | CHW    | Jackson Jobe   | Sean Burke       |       50.64 |       49.36 |       113 | +119             |      -136 | +125             |
| 2026-09-19 | MIL    | BAL    | Robert Gasser  | Trevor Rogers    |       57.91 |       42.09 |      -114 | -112             |      -106 | n/a              |
| 2026-09-19 | PHI    | NYM    | Andrew Painter | Christian Scott  |       51.98 |       48.02 |       104 | +113             |      -125 | +132             |
| 2026-09-19 | BOS    | TBD    | Brayan Bello   | Freddy Peralta   |       44    |       56    |       125 | +156             |      -151 | -104             |
| 2026-09-19 | OAK    | CLE    | Jacob Lopez    | Tanner Bibee     |       36    |       64    |       155 | n/a              |      -188 | -144             |
| 2026-09-19 | CHC    | CIN    | Matthew Boyd   | Nick Lodolo      |       59.74 |       40.26 |      -143 | -121             |       119 | n/a              |
| 2026-09-19 | KCR    | PIT    | Noah Cameron   | Bubba Chandler   |       43.74 |       56.26 |      -102 | +158             |      -118 | -105             |
| 2026-09-19 | TOR    | TEX    | José Soriano   | Cal Quantrill    |       47.69 |       52.31 |      -109 | +134             |      -110 | +111             |
| 2026-09-19 | ATL    | HOU    | Grant Holmes   | Hayden Wesneski  |       53.32 |       46.68 |       113 | +107             |      -137 | +140             |
| 2026-09-19 | WSN    | STL    | Andrew Alvarez | Michael McGreevy |       48.25 |       51.75 |      -107 | +131             |      -113 | +114             |
| 2026-09-19 | NYY    | ARI    | Cam Schlittler | Brandon Pfaadt   |       57.78 |       42.22 |      -168 | -112             |       138 | n/a              |
| 2026-09-19 | SEA    | COL    | Bryce Miller   | Jose Quintana    |       56.47 |       43.53 |      -155 | -106             |       128 | +160             |
| 2026-09-19 | FLA    | SDP    | Eury Pérez     | Casey Mize       |       41.13 |       58.87 |       131 | n/a              |      -159 | -117             |
| 2026-09-19 | SFG    | LAD    | Yunior Marte   | Tarik Skubal     |       33.44 |       66.56 |       317 | n/a              |      -415 | -160             |
| 2026-09-19 | MIN    | ANA    | Joe Ryan       | Reid Detmers     |       45.12 |       54.88 |      -108 | +149             |      -111 | +101             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 41322 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1570 |              0 |              10 | 100.000%   | 100.000%       | 100.000%         | 64.193%    | 37.026%    | 25.204%  |
|  2 | LAD    |         1561 |              3 |              10 | 100.000%   | 100.000%       | 100.000%         | 61.940%    | 31.472%    | 20.067%  |
|  3 | NYY    |         1551 |              9 |              13 | 100.000%   | 11.091%        | 100.000%         | 35.572%    | 25.647%    | 12.064%  |
|  4 | CHC    |         1546 |             -2 |               1 | 99.739%    | <0.007%        | 99.739%          | 21.202%    | 10.055%    | 5.726%   |
|  5 | BOS    |         1540 |              0 |              -9 | 100.000%   | <0.007%        | 100.000%         | 21.107%    | 14.542%    | 6.127%   |
|  6 | SDP    |         1534 |              6 |              11 | 98.870%    | <0.007%        | 98.870%          | 16.475%    | 7.287%     | 3.739%   |
|  7 | TBD    |         1534 |              6 |              15 | 100.000%   | 88.909%        | 100.000%         | 43.330%    | 28.723%    | 11.604%  |
|  8 | PHI    |         1533 |              2 |               5 | 99.557%    | 0.298%         | 99.557%          | 16.473%    | 6.863%     | 3.514%   |
|  9 | ATL    |         1531 |             -3 |               2 | 100.000%   | 99.702%        | 100.000%         | 19.571%    | 7.255%     | 3.705%   |
| 10 | DET    |         1513 |              7 |              -9 | 0.269%     | 0.041%         | 0.269%           | 0.077%     | 0.024%     | 0.010%   |
| 11 | PIT    |         1507 |              3 |               5 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 12 | TOR    |         1506 |             -3 |               0 | 14.239%    | <0.007%        | 14.239%          | 3.565%     | 1.210%     | 0.339%   |
| 13 | ARI    |         1502 |             -7 |              -9 | 1.834%     | <0.007%        | 1.834%           | 0.145%     | 0.041%     | 0.012%   |
| 14 | CLE    |         1502 |              6 |              11 | 89.817%    | 61.972%        | 89.817%          | 38.183%    | 12.642%    | 3.654%   |
| 15 | BAL    |         1497 |              3 |               6 | 2.042%     | <0.007%        | 2.042%           | 0.467%     | 0.145%     | 0.031%   |
| 16 | NYM    |         1496 |             -9 |               1 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 17 | CHW    |         1496 |             -3 |             -12 | 79.505%    | 37.987%        | 79.505%          | 26.686%    | 8.114%     | 2.067%   |
| 18 | FLA    |         1494 |              2 |              -4 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 19 | TEX    |         1492 |              9 |              11 | 56.500%    | 49.695%        | 56.500%          | 15.365%    | 4.417%     | 1.060%   |
| 20 | HOU    |         1490 |             -5 |               0 | 57.417%    | 50.114%        | 57.417%          | 15.587%    | 4.518%     | 1.074%   |
| 21 | STL    |         1484 |             -8 |             -19 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 22 | SEA    |         1481 |             -1 |              -4 | 0.201%     | 0.191%         | 0.201%           | 0.058%     | 0.017%     | 0.002%   |
| 23 | WSN    |         1479 |              6 |              -4 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 24 | KCR    |         1473 |             -5 |               6 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 25 | SFG    |         1472 |              1 |               6 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 26 | MIN    |         1465 |             -6 |             -14 | 0.010%     | <0.007%        | 0.010%           | 0.002%     | <0.007%    | <0.007%  |
| 27 | CIN    |         1465 |              2 |              -6 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 28 | ANA    |         1451 |             -1 |              -4 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 29 | OAK    |         1424 |             -4 |               1 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |
| 30 | COL    |         1413 |             -5 |             -17 | <0.007%    | <0.007%        | <0.007%          | <0.007%    | <0.007%    | <0.007%  |