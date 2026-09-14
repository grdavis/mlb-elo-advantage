# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-14 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 3% of games and have a -100.0% ROI over the last 7 days. ROI is -44.46% over the last 30 days and -0.69% over the last 365.

| Date       | Away   | Home   | Away Pitcher    | Home Pitcher    |   Away WinP |   Home WinP |   Away ML | Away Threshold   |   Home ML | Home Threshold   |
|:-----------|:-------|:-------|:----------------|:----------------|------------:|------------:|----------:|:-----------------|----------:|:-----------------|
| 2026-09-14 | LAD    | CIN    | Tarik Skubal    | Nick Lodolo     |       63.06 |       36.94 |      -220 | -138             |       179 | n/a              |
| 2026-09-14 | CHW    | CLE    | Sean Newcomb    | Gavin Williams  |       46.41 |       53.59 |       129 | +141             |      -156 | +106             |
| 2026-09-14 | DET    | TOR    | Troy Melton     | José Soriano    |       46.46 |       53.54 |       114 | +141             |      -138 | +106             |
| 2026-09-14 | BAL    | NYM    | Brandon Young   | Jonah Tong      |       44.93 |       55.07 |       104 | +150             |      -125 | -100             |
| 2026-09-14 | ATL    | CHC    | Reynaldo López  | David Peterson  |       44.87 |       55.13 |       113 | +151             |      -136 | -101             |
| 2026-09-14 | NYY    | MIN    | Will Warren     | Dean Kremer     |       59.06 |       40.94 |      -126 | -118             |       105 | n/a              |
| 2026-09-14 | SFG    | STL    | Landen Roupp    | Quinn Mathews   |       44.54 |       55.46 |       119 | +153             |      -144 | -102             |
| 2026-09-14 | SDP    | COL    | Casey Mize      | Tomoyuki Sugano |       64.49 |       35.51 |      -207 | -147             |       169 | n/a              |
| 2026-09-14 | SEA    | ANA    | Kade Anderson   | Reid Detmers    |       47.63 |       52.37 |      -114 | +135             |      -106 | +111             |
| 2026-09-14 | FLA    | ARI    | Sandy Alcantara | Corbin Burnes   |       41.96 |       58.04 |       110 | n/a              |      -132 | -113             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 27027 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1568 |              7 |              15 | 100.000%   | 100.000%       | 100.000%         | 63.873%    | 37.973%    | 25.752%  |
|  2 | LAD    |         1556 |              3 |               6 | 100.000%   | 99.996%        | 100.000%         | 58.882%    | 28.793%    | 18.293%  |
|  3 | CHC    |         1546 |              0 |              -4 | 97.824%    | <0.011%        | 97.824%          | 21.538%    | 10.834%    | 6.349%   |
|  4 | NYY    |         1544 |              2 |              14 | 100.000%   | 11.785%        | 100.000%         | 32.782%    | 22.936%    | 10.312%  |
|  5 | BOS    |         1542 |             -5 |              -5 | 99.989%    | 0.019%         | 99.989%          | 22.714%    | 15.640%    | 6.949%   |
|  6 | PHI    |         1534 |             -4 |              12 | 96.441%    | 1.717%         | 96.441%          | 17.053%    | 7.433%     | 3.959%   |
|  7 | TBD    |         1532 |             12 |               4 | 100.000%   | 88.197%        | 100.000%         | 44.581%    | 29.382%    | 12.099%  |
|  8 | ATL    |         1530 |             -3 |              -2 | 100.000%   | 98.283%        | 100.000%         | 22.744%    | 8.762%     | 4.721%   |
|  9 | SDP    |         1530 |              9 |               9 | 87.091%    | 0.004%         | 87.091%          | 13.934%    | 5.524%     | 2.779%   |
| 10 | TOR    |         1513 |              3 |              11 | 46.546%    | <0.011%        | 46.546%          | 13.272%    | 5.062%     | 1.554%   |
| 11 | DET    |         1509 |              7 |             -17 | 0.847%     | 0.170%         | 0.847%           | 0.233%     | 0.067%     | 0.022%   |
| 12 | ARI    |         1508 |             -2 |              -9 | 18.537%    | <0.011%        | 18.537%          | 1.965%     | 0.677%     | 0.266%   |
| 13 | PIT    |         1506 |              5 |               9 | 0.100%     | <0.011%        | 0.100%           | 0.011%     | 0.004%     | 0.004%   |
| 14 | NYM    |         1503 |              6 |              10 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 15 | CLE    |         1500 |              5 |               9 | 66.289%    | 41.555%        | 66.289%          | 26.115%    | 8.451%     | 2.346%   |
| 16 | CHW    |         1497 |            -10 |             -10 | 77.152%    | 58.234%        | 77.152%          | 31.402%    | 9.864%     | 2.546%   |
| 17 | FLA    |         1494 |             -2 |              -5 | 0.007%     | <0.011%        | 0.007%           | <0.011%    | <0.011%    | <0.011%  |
| 18 | STL    |         1493 |              0 |              -4 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 19 | HOU    |         1491 |             -5 |             -10 | 74.544%    | 71.495%        | 74.544%          | 20.831%    | 6.198%     | 1.532%   |
| 20 | BAL    |         1491 |             -3 |              -3 | 1.032%     | <0.011%        | 1.032%           | 0.196%     | 0.070%     | 0.019%   |
| 21 | TEX    |         1484 |              2 |               2 | 31.883%    | 27.069%        | 31.883%          | 7.515%     | 2.235%     | 0.481%   |
| 22 | SEA    |         1480 |              3 |              -8 | 1.528%     | 1.436%         | 1.528%           | 0.326%     | 0.089%     | 0.019%   |
| 23 | KCR    |         1477 |             -1 |              20 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 24 | WSN    |         1474 |              0 |             -11 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 25 | SFG    |         1469 |             -2 |              -5 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 26 | MIN    |         1467 |             -6 |              -7 | 0.189%     | 0.041%         | 0.189%           | 0.033%     | 0.007%     | <0.011%  |
| 27 | CIN    |         1465 |             -9 |             -12 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 28 | ANA    |         1451 |              3 |               3 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 29 | OAK    |         1430 |             -2 |               2 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |
| 30 | COL    |         1416 |            -10 |             -14 | <0.011%    | <0.011%        | <0.011%          | <0.011%    | <0.011%    | <0.011%  |