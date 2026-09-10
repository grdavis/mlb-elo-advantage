# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-10 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 5% of games and have a -100.0% ROI over the last 7 days. ROI is -24.63% over the last 30 days and 1.95% over the last 365.

| Date       | Away   | Home   | Away Pitcher    | Home Pitcher   |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   |   Home Threshold |
|:-----------|:-------|:-------|:----------------|:---------------|------------:|------------:|:----------|:-----------------|:----------|-----------------:|
| 2026-09-10 | TBD    | ATL    | Nick Martinez   | Martín Pérez   |       47.85 |       52.15 | NA        | +133             | NA        |             +112 |
| 2026-09-10 | HOU    | PHI    | Cristian Javier | Zack Wheeler   |       40.96 |       59.04 | NA        | n/a              | NA        |             -118 |
| 2026-09-10 | TEX    | SEA    | Jacob deGrom    | Logan Gilbert  |       50.83 |       49.17 | 105       | +118             | -126      |             +126 |
| 2026-09-10 | COL    | NYY    | Ryan Feltner    | Max Fried      |       27.32 |       72.68 | 239       | n/a              | -301      |             -209 |
| 2026-09-10 | PIT    | CHW    | Jared Jones     | Hagen Smith    |       47.51 |       52.49 | -110      | +135             | -110      |             +111 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 21276 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1564 |              0 |              12 | 100.000%   | 99.991%        | 100.000%         | 63.283%    | 35.867%    | 23.778%  |
|  2 | LAD    |         1558 |             10 |               7 | 100.000%   | 100.000%       | 100.000%         | 61.760%    | 32.191%    | 20.601%  |
|  3 | NYY    |         1545 |              3 |              11 | 100.000%   | 17.273%        | 100.000%         | 35.937%    | 25.052%    | 11.807%  |
|  4 | CHC    |         1543 |             -4 |             -11 | 95.850%    | 0.009%         | 95.850%          | 20.587%    | 9.992%     | 5.885%   |
|  5 | BOS    |         1540 |              0 |              -7 | 99.939%    | 0.165%         | 99.939%          | 23.501%    | 15.872%    | 7.163%   |
|  6 | PHI    |         1538 |             -1 |              20 | 98.214%    | 17.748%        | 98.214%          | 20.079%    | 9.245%     | 5.259%   |
|  7 | TBD    |         1527 |              8 |              -3 | 100.000%   | 82.563%        | 100.000%         | 40.680%    | 25.376%    | 9.969%   |
|  8 | ATL    |         1526 |             -6 |             -11 | 99.953%    | 82.252%        | 99.953%          | 20.008%    | 7.577%     | 3.718%   |
|  9 | SDP    |         1525 |              5 |               4 | 63.729%    | <0.014%        | 63.729%          | 9.494%     | 3.511%     | 1.758%   |
| 10 | TOR    |         1510 |             -1 |              11 | 41.065%    | <0.014%        | 41.065%          | 10.321%    | 3.779%     | 1.175%   |
| 11 | ARI    |         1509 |              2 |              -4 | 41.704%    | <0.014%        | 41.704%          | 4.747%     | 1.612%     | 0.757%   |
| 12 | PIT    |         1507 |              4 |              13 | 0.461%     | <0.014%        | 0.461%           | 0.042%     | 0.005%     | <0.014%  |
| 13 | DET    |         1503 |              2 |             -28 | 0.291%     | 0.099%         | 0.291%           | 0.085%     | 0.024%     | 0.005%   |
| 14 | CHW    |         1502 |             -2 |               0 | 89.570%    | 77.228%        | 89.570%          | 40.774%    | 14.505%    | 4.094%   |
| 15 | NYM    |         1501 |              8 |              10 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 16 | HOU    |         1496 |             -2 |              -5 | 82.431%    | 78.845%        | 82.431%          | 25.371%    | 8.394%     | 2.233%   |
| 17 | CLE    |         1495 |             -1 |               6 | 53.586%    | 22.462%        | 53.586%          | 16.084%    | 4.968%     | 1.340%   |
| 18 | BAL    |         1494 |             -4 |               2 | 3.276%     | <0.014%        | 3.276%           | 0.573%     | 0.169%     | 0.047%   |
| 19 | FLA    |         1493 |             -6 |              -8 | 0.047%     | <0.014%        | 0.047%           | <0.014%    | <0.014%    | <0.014%  |
| 20 | STL    |         1490 |             -5 |              -1 | 0.042%     | <0.014%        | 0.042%           | <0.014%    | <0.014%    | <0.014%  |
| 21 | TEX    |         1484 |              0 |              -7 | 26.659%    | 19.524%        | 26.659%          | 6.063%     | 1.706%     | 0.400%   |
| 22 | KCR    |         1478 |             -3 |              20 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 23 | SEA    |         1475 |             -6 |             -12 | 1.918%     | 1.631%         | 1.918%           | 0.400%     | 0.085%     | 0.014%   |
| 24 | SFG    |         1474 |              4 |              -2 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 25 | MIN    |         1472 |             -5 |              -7 | 1.264%     | 0.212%         | 1.264%           | 0.212%     | 0.071%     | <0.014%  |
| 26 | WSN    |         1470 |             -9 |             -17 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 27 | CIN    |         1469 |             -2 |             -10 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 28 | ANA    |         1455 |              8 |              13 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 29 | OAK    |         1433 |              6 |               5 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |
| 30 | COL    |         1423 |             -3 |              -4 | <0.014%    | <0.014%        | <0.014%          | <0.014%    | <0.014%    | <0.014%  |