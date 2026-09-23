# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-23 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 8% of games and have a -1.14% ROI over the last 7 days. ROI is -28.04% over the last 30 days and 1.27% over the last 365.

| Date       | Away   | Home   | Away Pitcher       | Home Pitcher       |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:-------------------|:-------------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-23 | WSN    | DET    | Richard Lovelady   | Framber Valdez     |       43.36 |       56.64 | NA        | +161             | NA        | -107             |
| 2026-09-23 | TOR    | BAL    | Max Scherzer       | Chris Bassitt      |       48.26 |       51.74 | NA        | +131             | NA        | +114             |
| 2026-09-23 | MIN    | SFG    | Connor Prielipp    | Cesar Perdomo      |       48.17 |       51.83 | -162      | +132             | 134       | +114             |
| 2026-09-23 | TOR    | BAL    | Max Scherzer       | Chris Bassitt      |       48.26 |       51.74 | 101       | +131             | -122      | +114             |
| 2026-09-23 | MIL    | PHI    | Logan Henderson    | Aaron Nola         |       53    |       47    | -129      | +108             | 107       | +138             |
| 2026-09-23 | STL    | PIT    | Matthew Liberatore | Lake Bachar        |       41.2  |       58.8  | 118       | n/a              | -143      | -116             |
| 2026-09-23 | TBD    | NYY    | Mason Englert      | Gerrit Cole        |       45.5  |       54.5  | 114       | +147             | -138      | +102             |
| 2026-09-23 | CLE    | BOS    | Foster Griffin     | Sonny Gray         |       41.32 |       58.68 | 113       | n/a              | -137      | -116             |
| 2026-09-23 | CIN    | ATL    | Andrew Abbott      | Chris Sale         |       33.26 |       66.74 | 222       | n/a              | -277      | -161             |
| 2026-09-23 | FLA    | CHC    | Ryan Gusto         | Kevin Gausman      |       39.8  |       60.2  | 149       | n/a              | -181      | -123             |
| 2026-09-23 | CHW    | KCR    | Bryan Hudson       | Seth Lugo          |       55.44 |       44.56 | -118      | -102             | -102      | +153             |
| 2026-09-23 | NYM    | TEX    | Nolan McLean       | Cody Bradford      |       50.32 |       49.68 | -108      | +121             | -111      | +124             |
| 2026-09-23 | ARI    | COL    | Merrill Kelly      | Mason Adams        |       62.56 |       37.44 | -162      | -136             | 134       | n/a              |
| 2026-09-23 | ANA    | OAK    | Walbert Ureña      | Jeffrey Springs    |       53.92 |       46.08 | -131      | +104             | 109       | +143             |
| 2026-09-23 | SDP    | LAD    | Robbie Ray         | Yoshinobu Yamamoto |       38.97 |       61.03 | 183       | n/a              | -224      | -127             |
| 2026-09-23 | HOU    | SEA    | Ethan Pecko        | George Kirby       |       49.48 |       50.52 | 105       | +125             | -127      | +120             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1571 |              0 |              10 | 100.000%   | 100.000%       | 100.000%         | 64.474%    | 36.360%    | 24.684%  |
|  2 | LAD    |         1568 |             12 |              13 | 100.000%   | 100.000%       | 100.000%         | 64.104%    | 33.906%    | 22.690%  |
|  3 | CHC    |         1546 |             -2 |              -5 | 99.708%    | <0.006%        | 99.708%          | 20.884%    | 9.784%     | 5.748%   |
|  4 | NYY    |         1544 |             -3 |               4 | 100.000%   | <0.006%        | 100.000%         | 29.690%    | 20.276%    | 8.728%   |
|  5 | TBD    |         1540 |              6 |              20 | 100.000%   | 100.000%       | 100.000%         | 50.768%    | 33.770%    | 14.164%  |
|  6 | BOS    |         1535 |             -2 |             -18 | 100.000%   | <0.006%        | 100.000%         | 20.154%    | 12.832%    | 4.996%   |
|  7 | PHI    |         1534 |              1 |               4 | 99.588%    | 0.018%         | 99.588%          | 16.132%    | 6.616%     | 3.476%   |
|  8 | SDP    |         1534 |              6 |               7 | 97.654%    | <0.006%        | 97.654%          | 16.026%    | 6.462%     | 3.342%   |
|  9 | ATL    |         1531 |              3 |               3 | 100.000%   | 99.982%        | 100.000%         | 18.098%    | 6.794%     | 3.520%   |
| 10 | PIT    |         1511 |              9 |              11 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 11 | ARI    |         1509 |              4 |               1 | 3.050%     | <0.006%        | 3.050%           | 0.282%     | 0.078%     | 0.030%   |
| 12 | DET    |         1508 |             -4 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 13 | CLE    |         1506 |              6 |               6 | 99.836%    | 58.122%        | 99.836%          | 42.210%    | 14.864%    | 4.020%   |
| 14 | TOR    |         1505 |             -4 |               1 | 0.404%     | <0.006%        | 0.404%           | 0.122%     | 0.032%     | 0.004%   |
| 15 | CHW    |         1503 |              7 |               1 | 99.400%    | 41.878%        | 99.400%          | 35.282%    | 11.988%    | 3.268%   |
| 16 | NYM    |         1500 |              4 |               4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | BAL    |         1495 |             -3 |               3 | 0.012%     | <0.006%        | 0.012%           | 0.006%     | <0.006%    | <0.006%  |
| 18 | FLA    |         1495 |             -2 |              -7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 19 | HOU    |         1490 |              0 |               2 | 54.680%    | 54.510%        | 54.680%          | 11.960%    | 3.490%     | 0.742%   |
| 20 | TEX    |         1489 |             -1 |               3 | 45.666%    | 45.488%        | 45.666%          | 9.808%     | 2.748%     | 0.588%   |
| 21 | STL    |         1482 |             -7 |             -15 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | WSN    |         1479 |              5 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | SEA    |         1476 |             -4 |             -10 | 0.002%     | 0.002%         | 0.002%           | <0.006%    | <0.006%    | <0.006%  |
| 24 | SFG    |         1469 |             -5 |               3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | KCR    |         1469 |             -9 |              -5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | MIN    |         1467 |              3 |              -8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | CIN    |         1464 |             -1 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1446 |             -6 |              -7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1423 |             -5 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1412 |             -5 |              -7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |