# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-16 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 3% of games and have a -18.67% ROI over the last 7 days. ROI is -37.84% over the last 30 days and -0.07% over the last 365.

| Date       | Away   | Home   | Away Pitcher    | Home Pitcher       |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:----------------|:-------------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-16 | CHW    | CLE    | Anthony Kay     | Parker Messick     |       44.31 |       55.69 | NA        | +154             | NA        | -103             |
| 2026-09-16 | SFG    | STL    | Anthony Molina  | Matthew Liberatore |       43.66 |       56.34 | NA        | +159             | NA        | -106             |
| 2026-09-16 | NYY    | MIN    | Carlos Rodón    | Zebby Matthews     |       62.19 |       37.81 | NA        | -134             | NA        | n/a              |
| 2026-09-16 | DET    | TOR    | Keider Montero  | Max Scherzer       |       48.45 |       51.55 | 109       | +130             | -132      | +115             |
| 2026-09-16 | LAD    | CIN    | Blake Snell     | Andrew Abbott      |       66.36 |       33.64 | -219      | -159             | 178       | n/a              |
| 2026-09-16 | MIL    | PIT    | Logan Henderson | Jared Jones        |       57.72 |       42.28 | -126      | -111             | 104       | n/a              |
| 2026-09-16 | OAK    | TBD    | Brady Basso     | Nick Martinez      |       32.13 |       67.87 | 153       | n/a              | -186      | -169             |
| 2026-09-16 | PHI    | WSN    | Zack Wheeler    | Jared Simpson      |       54.57 |       45.43 | -199      | +102             | 163       | +147             |
| 2026-09-16 | BAL    | NYM    | Chris Bassitt   | Robert Stock       |       46.17 |       53.83 | 112       | +143             | -135      | +105             |
| 2026-09-16 | ATL    | CHC    | JR Ritchie      | Shota Imanaga      |       44.13 |       55.87 | 135       | +156             | -164      | -104             |
| 2026-09-16 | BOS    | TEX    | Jake Bennett    | MacKenzie Gore     |       54.93 |       45.07 | -112      | +100             | -108      | +150             |
| 2026-09-16 | KCR    | HOU    | Daniel Lynch IV | Cristian Javier    |       43.49 |       56.51 | 138       | +160             | -167      | -106             |
| 2026-09-16 | SDP    | COL    | Robbie Ray      | Mason Adams        |       61.62 |       38.38 | -172      | -131             | 142       | n/a              |
| 2026-09-16 | SEA    | ANA    | George Kirby    | Yusei Kikuchi      |       48.59 |       51.41 | -145      | +129             | 120       | +115             |
| 2026-09-16 | FLA    | ARI    | Ryan Gusto      | Merrill Kelly      |       45.18 |       54.82 | 113       | +149             | -137      | +101             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 31250 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1570 |              6 |              14 | 100.000%   | 100.000%       | 100.000%         | 64.854%    | 38.195%    | 26.157%  |
|  2 | LAD    |         1560 |              2 |              11 | 100.000%   | 100.000%       | 100.000%         | 62.141%    | 31.210%    | 20.122%  |
|  3 | NYY    |         1549 |              4 |              18 | 100.000%   | 14.848%        | 100.000%         | 36.515%    | 26.208%    | 12.029%  |
|  4 | CHC    |         1546 |              3 |              -1 | 98.982%    | <0.010%        | 98.982%          | 22.093%    | 10.691%    | 6.022%   |
|  5 | BOS    |         1540 |              0 |              -8 | 99.978%    | 0.003%         | 99.978%          | 21.318%    | 14.579%    | 6.070%   |
|  6 | TBD    |         1533 |              6 |               8 | 100.000%   | 85.149%        | 100.000%         | 42.269%    | 27.872%    | 11.238%  |
|  7 | ATL    |         1531 |              5 |              -1 | 100.000%   | 99.427%        | 100.000%         | 21.024%    | 8.150%     | 4.240%   |
|  8 | PHI    |         1531 |             -7 |               6 | 95.949%    | 0.573%         | 95.949%          | 15.162%    | 6.186%     | 3.248%   |
|  9 | SDP    |         1526 |              1 |               4 | 87.866%    | <0.010%        | 87.866%          | 13.139%    | 5.088%     | 2.531%   |
| 10 | DET    |         1515 |             12 |             -11 | 2.208%     | 0.611%         | 2.208%           | 0.698%     | 0.266%     | 0.077%   |
| 11 | ARI    |         1507 |             -2 |              -5 | 17.158%    | <0.010%        | 17.158%          | 1.584%     | 0.480%     | 0.170%   |
| 12 | TOR    |         1506 |             -4 |               5 | 21.795%    | <0.010%        | 21.795%          | 5.498%     | 1.955%     | 0.573%   |
| 13 | PIT    |         1504 |             -3 |               6 | 0.038%     | <0.010%        | 0.038%           | 0.003%     | <0.010%    | <0.010%  |
| 14 | NYM    |         1500 |             -1 |               5 | <0.010%    | <0.010%        | <0.010%          | <0.010%    | <0.010%    | <0.010%  |
| 15 | CHW    |         1498 |             -4 |             -10 | 84.630%    | 61.658%        | 84.630%          | 34.547%    | 11.187%    | 3.114%   |
| 16 | CLE    |         1498 |              3 |              10 | 72.790%    | 37.731%        | 72.790%          | 25.152%    | 7.821%     | 2.045%   |
| 17 | FLA    |         1495 |              2 |              -7 | 0.006%     | <0.010%        | 0.006%           | <0.010%    | <0.010%    | <0.010%  |
| 18 | BAL    |         1494 |              0 |              -3 | 3.267%     | <0.010%        | 3.267%           | 0.701%     | 0.214%     | 0.048%   |
| 19 | HOU    |         1493 |             -3 |              -6 | 76.973%    | 71.277%        | 76.973%          | 23.443%    | 7.043%     | 1.670%   |
| 20 | STL    |         1490 |              0 |             -11 | <0.010%    | <0.010%        | <0.010%          | <0.010%    | <0.010%    | <0.010%  |
| 21 | TEX    |         1487 |              3 |               7 | 38.230%    | 28.630%        | 38.230%          | 9.834%     | 2.851%     | 0.646%   |
| 22 | SEA    |         1477 |              2 |             -12 | 0.099%     | 0.093%         | 0.099%           | 0.019%     | <0.010%    | <0.010%  |
| 23 | WSN    |         1476 |              6 |              -8 | <0.010%    | <0.010%        | <0.010%          | <0.010%    | <0.010%    | <0.010%  |
| 24 | KCR    |         1475 |             -3 |              13 | <0.010%    | <0.010%        | <0.010%          | <0.010%    | <0.010%    | <0.010%  |
| 25 | SFG    |         1472 |             -2 |               2 | <0.010%    | <0.010%        | <0.010%          | <0.010%    | <0.010%    | <0.010%  |
| 26 | MIN    |         1462 |            -10 |             -12 | 0.029%     | <0.010%        | 0.029%           | 0.006%     | 0.003%     | <0.010%  |
| 27 | CIN    |         1462 |             -7 |             -12 | <0.010%    | <0.010%        | <0.010%          | <0.010%    | <0.010%    | <0.010%  |
| 28 | ANA    |         1455 |              0 |               9 | <0.010%    | <0.010%        | <0.010%          | <0.010%    | <0.010%    | <0.010%  |
| 29 | OAK    |         1429 |             -4 |               1 | <0.010%    | <0.010%        | <0.010%          | <0.010%    | <0.010%    | <0.010%  |
| 30 | COL    |         1419 |             -4 |             -13 | <0.010%    | <0.010%        | <0.010%          | <0.010%    | <0.010%    | <0.010%  |