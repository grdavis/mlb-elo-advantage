# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-24 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 7% of games and have a 15.33% ROI over the last 7 days. ROI is -31.48% over the last 30 days and 1.4% over the last 365.

| Date       | Away   | Home   | Away Pitcher      | Home Pitcher   |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:------------------|:---------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-24 | STL    | PIT    | Kyle Leahy        | Paul Skenes    |       43.81 |       56.19 | NA        | +158             | NA        | -105             |
| 2026-09-24 | CHW    | KCR    | David Sandlin     | Randy Dobnak   |       52.65 |       47.35 | NA        | +110             | NA        | +136             |
| 2026-09-24 | FLA    | CHC    | Tyler Phillips    | Matthew Boyd   |       40.5  |       59.5  | NA        | n/a              | NA        | -120             |
| 2026-09-24 | NYM    | TEX    | Zac Thornton      | Kumar Rocker   |       48.99 |       51.01 | 113       | +127             | -136      | +117             |
| 2026-09-24 | ARI    | COL    | Eduardo Rodriguez | Tanner Gordon  |       63.49 |       36.51 | -199      | -141             | 163       | n/a              |
| 2026-09-24 | MIL    | PHI    | Shane Drohan      | Andrew Painter |       51.4  |       48.6  | 113       | +116             | -136      | +129             |
| 2026-09-24 | CLE    | BOS    | Daniel Espino     | Ranger Suarez  |       41.99 |       58.01 | 109       | n/a              | -131      | -113             |
| 2026-09-24 | TBD    | NYY    | Ian Seymour       | Cam Schlittler |       41.34 |       58.66 | 129       | n/a              | -156      | -116             |
| 2026-09-24 | CIN    | ATL    | Brady Singer      | Tyler Mahle    |       31.44 |       68.56 | 188       | n/a              | -231      | -174             |
| 2026-09-24 | HOU    | OAK    | Peter Lambert     | Mason Barnett  |       59.61 |       40.39 | -171      | -120             | 141       | n/a              |
| 2026-09-24 | ANA    | SEA    | Grayson Rodriguez | Bryan Woo      |       40.2  |       59.8  | 187       | n/a              | -231      | -121             |
| 2026-09-24 | SDP    | LAD    | Nick Pivetta      | Tyler Glasnow  |       43.58 |       56.42 | 144       | +159             | -175      | -106             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1573 |              5 |              14 | 100.000%   | 100.000%       | 100.000%         | 65.366%    | 38.302%    | 26.574%  |
|  2 | LAD    |         1564 |              5 |              10 | 100.000%   | 100.000%       | 100.000%         | 63.200%    | 31.960%    | 21.052%  |
|  3 | NYY    |         1548 |              1 |              11 | 100.000%   | <0.006%        | 100.000%         | 31.260%    | 21.914%    | 9.684%   |
|  4 | CHC    |         1545 |             -3 |              -5 | 98.654%    | <0.006%        | 98.654%          | 18.912%    | 8.620%     | 4.884%   |
|  5 | SDP    |         1537 |              6 |              11 | 98.902%    | <0.006%        | 98.902%          | 17.698%    | 7.626%     | 4.058%   |
|  6 | TBD    |         1536 |              0 |              18 | 100.000%   | 100.000%       | 100.000%         | 48.800%    | 31.966%    | 13.034%  |
|  7 | BOS    |         1536 |             -2 |             -20 | 100.000%   | <0.006%        | 100.000%         | 19.980%    | 13.160%    | 5.130%   |
|  8 | ATL    |         1532 |              4 |               3 | 100.000%   | 100.000%       | 100.000%         | 19.282%    | 7.306%     | 3.776%   |
|  9 | PHI    |         1532 |             -4 |               4 | 98.674%    | <0.006%        | 98.674%          | 15.126%    | 6.050%     | 3.162%   |
| 10 | ARI    |         1510 |              5 |               0 | 3.770%     | <0.006%        | 3.770%           | 0.416%     | 0.136%     | 0.072%   |
| 11 | PIT    |         1508 |              3 |               7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 12 | DET    |         1506 |             -5 |              -8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 13 | CLE    |         1505 |              5 |               3 | 99.998%    | 62.900%        | 99.998%          | 44.022%    | 15.436%    | 4.256%   |
| 14 | NYM    |         1503 |              9 |               6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 15 | CHW    |         1502 |              4 |              -3 | 99.840%    | 37.100%        | 99.840%          | 33.984%    | 11.398%    | 3.034%   |
| 16 | TOR    |         1501 |             -8 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | BAL    |         1499 |              1 |               1 | 0.048%     | <0.006%        | 0.048%           | 0.006%     | <0.006%    | <0.006%  |
| 18 | FLA    |         1497 |              0 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 19 | HOU    |         1489 |             -4 |              -2 | 56.878%    | 56.818%        | 56.878%          | 12.844%    | 3.648%     | 0.784%   |
| 20 | TEX    |         1486 |             -2 |               2 | 43.226%    | 43.172%        | 43.226%          | 9.104%     | 2.478%     | 0.500%   |
| 21 | STL    |         1485 |             -4 |              -6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | WSN    |         1481 |              7 |               2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | SEA    |         1477 |             -3 |             -12 | 0.010%     | 0.010%         | 0.010%           | <0.006%    | <0.006%    | <0.006%  |
| 24 | KCR    |         1470 |             -5 |              -7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | MIN    |         1469 |              7 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | SFG    |         1467 |             -7 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | CIN    |         1463 |              0 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1443 |            -10 |              -9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1426 |              0 |               1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1411 |             -3 |             -11 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |