# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-11 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 5% of games and have a -100.0% ROI over the last 7 days. ROI is -35.68% over the last 30 days and 0.82% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher       |   Away WinP |   Home WinP |   Away ML | Away Threshold   |   Home ML | Home Threshold   |
|:-----------|:-------|:-------|:---------------|:-------------------|------------:|------------:|----------:|:-----------------|----------:|:-----------------|
| 2026-09-11 | PIT    | CHC    | Wilber Dotel   | Shota Imanaga      |       43.61 |       56.39 |       159 | +159             |      -194 | -106             |
| 2026-09-11 | COL    | DET    | Mason Adams    | Framber Valdez     |       34.89 |       65.11 |       154 | n/a              |      -187 | -151             |
| 2026-09-11 | ANA    | WSN    | Yusei Kikuchi  | Cade Cavalli       |       42.57 |       57.43 |       129 | n/a              |      -156 | -110             |
| 2026-09-11 | NYM    | NYY    | Nolan McLean   | Carlos Rodón       |       41.01 |       58.99 |       114 | n/a              |      -137 | -117             |
| 2026-09-11 | BAL    | TOR    | Chris Bassitt  | Max Scherzer       |       43.93 |       56.07 |       114 | +157             |      -138 | -104             |
| 2026-09-11 | KCR    | BOS    | Seth Lugo      | Sonny Gray         |       35.19 |       64.81 |       168 | n/a              |      -206 | -149             |
| 2026-09-11 | LAD    | FLA    |                | Ryan Gusto         |       57.92 |       42.08 |      -208 | -112             |       170 | n/a              |
| 2026-09-11 | HOU    | TBD    | Miguel Ullola  | Drew Rasmussen     |       41.86 |       58.14 |       144 | n/a              |      -174 | -113             |
| 2026-09-11 | PHI    | ATL    | Aaron Nola     | Chris Sale         |       46.69 |       53.31 |       154 | +140             |      -187 | +107             |
| 2026-09-11 | CIN    | MIL    | Andrew Abbott  | Dustin May         |       33.35 |       66.65 |       169 | n/a              |      -206 | -161             |
| 2026-09-11 | CLE    | MIN    | Parker Messick | Taj Bradley        |       52.07 |       47.93 |      -118 | +112             |      -102 | +133             |
| 2026-09-11 | CHW    | STL    | Anthony Kay    | Matthew Liberatore |       48.36 |       51.64 |      -114 | +131             |      -106 | +114             |
| 2026-09-11 | TEX    | ARI    | MacKenzie Gore |                    |       41.74 |       58.26 |      -102 | n/a              |      -118 | -114             |
| 2026-09-11 | SEA    | OAK    | George Kirby   | Jeffrey Springs    |       53.75 |       46.25 |      -168 | +105             |       138 | +142             |
| 2026-09-11 | SDP    | SFG    | Robbie Ray     | Anthony Molina     |       55.33 |       44.67 |      -149 | -101             |       123 | +152             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 21739 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1564 |             -1 |              14 | 100.000%   | 99.991%        | 100.000%         | 63.352%    | 35.894%    | 24.063%  |
|  2 | LAD    |         1558 |              8 |               6 | 100.000%   | 100.000%       | 100.000%         | 61.001%    | 31.533%    | 20.015%  |
|  3 | NYY    |         1547 |              6 |              11 | 100.000%   | 25.158%        | 100.000%         | 39.105%    | 27.467%    | 13.147%  |
|  4 | CHC    |         1543 |             -7 |             -14 | 95.814%    | 0.009%         | 95.814%          | 20.958%    | 10.005%    | 5.902%   |
|  5 | BOS    |         1540 |             -2 |              -4 | 99.926%    | 0.193%         | 99.926%          | 23.152%    | 15.543%    | 6.960%   |
|  6 | PHI    |         1537 |              1 |              22 | 97.203%    | 8.749%         | 97.203%          | 19.219%    | 8.979%     | 4.706%   |
|  7 | ATL    |         1528 |             -7 |             -11 | 99.995%    | 91.251%        | 99.995%          | 21.229%    | 8.404%     | 4.186%   |
|  8 | TBD    |         1526 |              6 |              -6 | 100.000%   | 74.649%        | 100.000%         | 37.872%    | 23.566%    | 9.209%   |
|  9 | SDP    |         1525 |              4 |               3 | 64.124%    | <0.014%        | 64.124%          | 9.393%     | 3.643%     | 1.762%   |
| 10 | TOR    |         1510 |             -4 |               9 | 43.245%    | <0.014%        | 43.245%          | 11.109%    | 4.131%     | 1.279%   |
| 11 | ARI    |         1509 |              4 |              -2 | 42.003%    | <0.014%        | 42.003%          | 4.724%     | 1.523%     | 0.695%   |
| 12 | PIT    |         1509 |              5 |              18 | 0.777%     | <0.014%        | 0.777%           | 0.110%     | 0.018%     | 0.005%   |
| 13 | DET    |         1503 |              4 |             -26 | 0.359%     | 0.106%         | 0.359%           | 0.087%     | 0.041%     | 0.023%   |
| 14 | NYM    |         1501 |              6 |              12 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 15 | CHW    |         1500 |             -6 |              -4 | 85.390%    | 71.199%        | 85.390%          | 35.011%    | 11.905%    | 3.376%   |
| 16 | HOU    |         1498 |             -1 |              -4 | 90.975%    | 89.057%        | 90.975%          | 31.276%    | 10.207%    | 2.783%   |
| 17 | CLE    |         1495 |             -4 |               4 | 56.603%    | 28.460%        | 56.603%          | 17.637%    | 5.787%     | 1.610%   |
| 18 | BAL    |         1494 |             -3 |               4 | 3.648%     | <0.014%        | 3.648%           | 0.759%     | 0.230%     | 0.069%   |
| 19 | FLA    |         1493 |             -3 |             -11 | 0.055%     | <0.014%        | 0.055%           | 0.009%     | <0.014%    | <0.014%  |
| 20 | STL    |         1490 |             -6 |              -5 | 0.028%     | <0.014%        | 0.028%           | 0.005%     | <0.014%    | <0.014%  |
| 21 | TEX    |         1483 |              0 |              -5 | 15.750%    | 9.048%         | 15.750%          | 3.243%     | 0.943%     | 0.170%   |
| 22 | KCR    |         1478 |              1 |              21 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 23 | SEA    |         1477 |             -2 |              -7 | 2.682%     | 1.895%         | 2.682%           | 0.515%     | 0.133%     | 0.028%   |
| 24 | SFG    |         1474 |              6 |               0 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 25 | MIN    |         1472 |             -3 |              -9 | 1.421%     | 0.235%         | 1.421%           | 0.235%     | 0.046%     | 0.014%   |
| 26 | WSN    |         1470 |             -7 |             -14 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 27 | CIN    |         1469 |              0 |              -8 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 28 | ANA    |         1455 |              9 |              11 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 29 | OAK    |         1433 |              4 |               7 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 30 | COL    |         1421 |             -4 |              -9 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |