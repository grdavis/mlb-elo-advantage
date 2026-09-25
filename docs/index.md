# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-25 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 7% of games and have a 15.33% ROI over the last 7 days. ROI is -39.43% over the last 30 days and 1.41% over the last 365.

| Date       | Away   | Home   | Away Pitcher     | Home Pitcher       |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:-----------------|:-------------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-25 | BAL    | NYY    | Trevor Rogers    | Brendan Beck       |       42.05 |       57.95 | -101      | n/a              | -119      | -113             |
| 2026-09-25 | CHC    | BOS    | Clay Holmes      | Alec Gamboa        |       47.23 |       52.77 | NA        | +137             | NA        | +109             |
| 2026-09-25 | BAL    | NYY    | Brandon Young    |                    |       38.46 |       61.54 | 102       | n/a              | -123      | -130             |
| 2026-09-25 | CHC    | BOS    | David Peterson   | Brayan Bello       |       50.41 |       49.59 | -115      | +120             | -105      | +124             |
| 2026-09-25 | TBD    | PHI    | Freddy Peralta   | Cristopher Sánchez |       49.64 |       50.36 | 147       | +124             | -179      | +120             |
| 2026-09-25 | PIT    | DET    | Bubba Chandler   | Jackson Jobe       |       46.74 |       53.26 | -110      | +140             | -109      | +107             |
| 2026-09-25 | NYM    | WSN    |                  | Andrew Alvarez     |       48.66 |       51.34 | -112      | +129             | -108      | +116             |
| 2026-09-25 | CIN    | TOR    | Nick Lodolo      |                    |       39.74 |       60.26 | 118       | n/a              | -143      | -123             |
| 2026-09-25 | ATL    | FLA    |                  | Eury Pérez         |       55.17 |       44.83 | -118      | -101             | -102      | +151             |
| 2026-09-25 | STL    | MIL    | Michael McGreevy | Robert Gasser      |       33.17 |       66.83 | 157       | n/a              | -192      | -162             |
| 2026-09-25 | CLE    | KCR    | Gavin Williams   | Noah Cameron       |       54.55 |       45.45 | -145      | +102             | 120       | +147             |
| 2026-09-25 | COL    | CHW    | Tomoyuki Sugano  | Sean Burke         |       32.58 |       67.42 | 198       | n/a              | -245      | -166             |
| 2026-09-25 | TEX    | MIN    | Jacob deGrom     | Joe Ryan           |       52.52 |       47.48 | -114      | +110             | -105      | +135             |
| 2026-09-25 | ARI    | SDP    | Brandon Pfaadt   |                    |       42.6  |       57.4  | 111       | n/a              | -133      | -110             |
| 2026-09-25 | HOU    | OAK    | Hunter Brown     | Jacob Lopez        |       61.03 |       38.97 | -185      | -127             | 152       | n/a              |
| 2026-09-25 | ANA    | SEA    | Reid Detmers     | Bryce Miller       |       45.12 |       54.88 | 102       | +149             | -123      | +100             |
| 2026-09-25 | LAD    | SFG    | Tarik Skubal     | Yunior Marte       |       62.46 |       37.54 | -336      | -135             | 264       | n/a              |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1575 |              5 |              13 | 100.000%   | 100.000%       | 100.000%         | 65.808%    | 40.190%    | 27.942%  |
|  2 | LAD    |         1562 |              1 |              10 | 100.000%   | 100.000%       | 100.000%         | 63.330%    | 30.730%    | 20.022%  |
|  3 | NYY    |         1549 |             -2 |               9 | 100.000%   | <0.006%        | 100.000%         | 33.098%    | 23.098%    | 10.408%  |
|  4 | CHC    |         1546 |              0 |              -2 | 99.614%    | <0.006%        | 99.614%          | 18.138%    | 8.574%     | 4.918%   |
|  5 | SDP    |         1539 |              5 |              11 | 100.000%   | <0.006%        | 100.000%         | 18.122%    | 8.150%     | 4.476%   |
|  6 | TBD    |         1535 |              1 |              15 | 100.000%   | 100.000%       | 100.000%         | 47.646%    | 30.568%    | 12.002%  |
|  7 | BOS    |         1534 |             -6 |             -19 | 100.000%   | <0.006%        | 100.000%         | 19.436%    | 12.280%    | 4.698%   |
|  8 | ATL    |         1530 |             -1 |              -1 | 100.000%   | 100.000%       | 100.000%         | 19.806%    | 7.100%     | 3.708%   |
|  9 | PHI    |         1529 |             -4 |              -2 | 96.024%    | <0.006%        | 96.024%          | 14.288%    | 5.094%     | 2.530%   |
| 10 | ARI    |         1512 |             10 |               0 | 4.362%     | <0.006%        | 4.362%           | 0.508%     | 0.162%     | 0.070%   |
| 11 | PIT    |         1509 |              2 |              10 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 12 | CLE    |         1507 |              5 |               4 | 100.000%   | 66.832%        | 100.000%         | 44.984%    | 15.870%    | 4.456%   |
| 13 | CHW    |         1506 |             10 |              -1 | 99.972%    | 33.168%        | 99.972%          | 33.856%    | 12.126%    | 3.418%   |
| 14 | DET    |         1506 |             -7 |              -6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 15 | TOR    |         1501 |             -5 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 16 | NYM    |         1501 |              5 |               7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | BAL    |         1499 |              2 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 18 | FLA    |         1496 |              2 |              -6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 19 | HOU    |         1490 |              0 |               2 | 56.458%    | 56.446%        | 56.458%          | 12.168%    | 3.592%     | 0.776%   |
| 20 | TEX    |         1488 |             -4 |               7 | 43.570%    | 43.554%        | 43.570%          | 8.812%     | 2.466%     | 0.576%   |
| 21 | STL    |         1484 |              0 |              -6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | WSN    |         1481 |              2 |               9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | SEA    |         1475 |             -6 |             -11 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 24 | MIN    |         1469 |              4 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | SFG    |         1467 |             -5 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | KCR    |         1466 |             -7 |              -9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | CIN    |         1465 |              0 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1445 |             -6 |              -5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1424 |              0 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1409 |             -4 |             -19 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |