# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-01 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 8% of games and have a 29.37% ROI over the last 7 days. ROI is -29.93% over the last 30 days and 1.93% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher   |   Away WinP |   Home WinP |   Away ML |   Away Threshold |   Home ML |   Home Threshold |
|:-----------|:-------|:-------|:---------------|:---------------|------------:|------------:|----------:|-----------------:|----------:|-----------------:|
| 2026-10-01 | PHI    | ATL    | Aaron Nola     | Ray Kerr       |       44.75 |       55.25 |      -111 |             +152 |      -108 |             -101 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1579 |              4 |              17 | 100.000%   | 100.000%       | 100.000%         | 62.800%    | 38.634%    | 26.660%  |
|  2 | LAD    |         1566 |              4 |              16 | 100.000%   | 100.000%       | 100.000%         | 65.142%    | 32.364%    | 20.878%  |
|  3 | NYY    |         1553 |              4 |              12 | 100.000%   | <0.006%        | 100.000%         | 53.680%    | 36.206%    | 16.326%  |
|  4 | SDP    |         1546 |              7 |              23 | 100.000%   | <0.006%        | 100.000%         | 37.200%    | 17.302%    | 9.376%   |
|  5 | TBD    |         1539 |              4 |              13 | 100.000%   | 100.000%       | 100.000%         | 46.320%    | 28.658%    | 11.336%  |
|  6 | CHC    |         1539 |             -7 |             -10 | 100.000%   | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
|  7 | ATL    |         1529 |             -1 |               1 | 100.000%   | 100.000%       | 54.278%          | 19.168%    | 6.586%     | 3.142%   |
|  8 | BOS    |         1527 |             -7 |             -16 | 100.000%   | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
|  9 | PHI    |         1525 |             -4 |             -15 | 100.000%   | <0.006%        | 45.722%          | 15.690%    | 5.114%     | 2.322%   |
| 10 | ARI    |         1512 |              0 |               6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 11 | CHW    |         1511 |              5 |               2 | 100.000%   | <0.006%        | 100.000%         | 48.490%    | 16.822%    | 4.846%   |
| 12 | CLE    |         1511 |              4 |               7 | 100.000%   | 100.000%       | 100.000%         | 51.510%    | 18.314%    | 5.114%   |
| 13 | PIT    |         1508 |             -1 |               5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 14 | DET    |         1507 |              1 |               9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 15 | BAL    |         1502 |              3 |               1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 16 | TOR    |         1501 |              0 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | NYM    |         1501 |              0 |              12 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 18 | FLA    |         1497 |              1 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 19 | HOU    |         1491 |              1 |              -2 | 100.000%   | 100.000%       | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 20 | TEX    |         1484 |             -4 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 21 | WSN    |         1481 |              0 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | STL    |         1480 |             -4 |             -13 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | SEA    |         1477 |              2 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 24 | MIN    |         1472 |              3 |              -9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | CIN    |         1464 |             -1 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | SFG    |         1463 |             -4 |              -7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | KCR    |         1463 |             -3 |             -17 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1443 |             -2 |              -6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1418 |             -6 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1409 |              0 |             -15 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |