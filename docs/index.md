# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-13 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 4% of games and have a -100.0% ROI over the last 7 days. ROI is -45.93% over the last 30 days and -0.04% over the last 365.

| Date       | Away   | Home   | Away Pitcher      | Home Pitcher      |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:------------------|:------------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-13 | COL    | DET    | Gabriel Hughes    | Jackson Jobe      |       35.09 |       64.91 | NA        | n/a              | NA        | -149             |
| 2026-09-13 | PHI    | ATL    | Andrew Painter    | Grant Holmes      |       45.25 |       54.75 | 109       | +148             | -131      | +101             |
| 2026-09-13 | NYM    | NYY    | Christian Scott   | Cam Schlittler    |       39.25 |       60.75 | 152       | n/a              | -185      | -126             |
| 2026-09-13 | ANA    | WSN    | Grayson Rodriguez | Riley Cornelio    |       41.37 |       58.63 | 114       | n/a              | -138      | -116             |
| 2026-09-13 | BAL    | TOR    | Trevor Rogers     | Dylan Cease       |       44.15 |       55.85 | 127       | +155             | -154      | -103             |
| 2026-09-13 | LAD    | FLA    | Emmet Sheehan     | Eury Pérez        |       57.77 |       42.23 | -137      | -112             | 114       | n/a              |
| 2026-09-13 | HOU    | TBD    | Hayden Wesneski   | Freddy Peralta    |       42.96 |       57.04 | 110       | +163             | -133      | -109             |
| 2026-09-13 | CIN    | MIL    | Chase Burns       | Robert Gasser     |       33.46 |       66.54 | 169       | n/a              | -206      | -160             |
| 2026-09-13 | CLE    | MIN    | Tanner Bibee      | Joe Ryan          |       50.54 |       49.46 | 104       | +120             | -126      | +125             |
| 2026-09-13 | CHW    | STL    | David Sandlin     | Michael McGreevy  |       48.96 |       51.04 | -117      | +127             | -103      | +117             |
| 2026-09-13 | PIT    | CHC    | Bubba Chandler    | Matthew Boyd      |       41.74 |       58.26 | 141       | n/a              | -171      | -114             |
| 2026-09-13 | KCR    | BOS    | Noah Cameron      | Payton Tolle      |       38.35 |       61.65 | 158       | n/a              | -192      | -131             |
| 2026-09-13 | SEA    | OAK    | Bryce Miller      | Jacob Lopez       |       53.03 |       46.97 | -134      | +108             | 111       | +138             |
| 2026-09-13 | TEX    | ARI    | Cal Quantrill     | Eduardo Rodriguez |       43.56 |       56.44 | 109       | +159             | -131      | -106             |
| 2026-09-13 | SDP    | SFG    | Nick Pivetta      | Logan Webb        |       57.71 |       42.29 | -131      | -111             | 109       | n/a              |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 25000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1570 |             10 |              20 | 100.000%   | 100.000%       | 100.000%         | 64.360%    | 38.508%    | 26.944%  |
|  2 | LAD    |         1558 |              6 |               6 | 100.000%   | 99.992%        | 100.000%         | 59.564%    | 29.396%    | 18.752%  |
|  3 | CHC    |         1548 |              1 |              -6 | 98.440%    | <0.012%        | 98.440%          | 23.188%    | 11.436%    | 6.800%   |
|  4 | NYY    |         1542 |              0 |              10 | 100.000%   | 12.496%        | 100.000%         | 33.756%    | 23.340%    | 10.340%  |
|  5 | BOS    |         1540 |             -6 |              -5 | 99.964%    | 0.040%         | 99.964%          | 23.916%    | 16.144%    | 6.916%   |
|  6 | ATL    |         1534 |             -1 |              -2 | 100.000%   | 99.180%        | 100.000%         | 23.692%    | 9.264%     | 5.304%   |
|  7 | PHI    |         1531 |             -6 |              13 | 91.196%    | 0.820%         | 91.196%          | 14.312%    | 5.808%     | 2.924%   |
|  8 | SDP    |         1528 |              8 |               4 | 78.976%    | 0.008%         | 78.976%          | 11.696%    | 4.560%     | 2.320%   |
|  9 | TBD    |         1528 |              8 |              -2 | 100.000%   | 87.464%        | 100.000%         | 42.480%    | 27.056%    | 10.432%  |
| 10 | TOR    |         1509 |             -3 |               9 | 40.876%    | <0.012%        | 40.876%          | 11.092%    | 4.124%     | 1.252%   |
| 11 | ARI    |         1509 |              1 |              -4 | 31.312%    | <0.012%        | 31.312%          | 3.164%     | 1.020%     | 0.440%   |
| 12 | DET    |         1506 |              5 |             -21 | 0.652%     | 0.140%         | 0.652%           | 0.176%     | 0.076%     | 0.020%   |
| 13 | NYM    |         1505 |             11 |              14 | <0.012%    | <0.012%        | <0.012%          | <0.012%    | <0.012%    | <0.012%  |
| 14 | PIT    |         1504 |              3 |               4 | 0.064%     | <0.012%        | 0.064%           | 0.016%     | 0.008%     | <0.012%  |
| 15 | CHW    |         1499 |             -8 |              -7 | 86.740%    | 71.212%        | 86.740%          | 37.292%    | 12.660%    | 3.464%   |
| 16 | CLE    |         1496 |             -1 |               9 | 59.296%    | 28.492%        | 59.296%          | 19.200%    | 6.400%     | 1.644%   |
| 17 | HOU    |         1495 |             -1 |              -9 | 83.680%    | 81.340%        | 83.680%          | 25.920%    | 8.436%     | 2.076%   |
| 18 | BAL    |         1494 |              2 |               2 | 3.092%     | <0.012%        | 3.092%           | 0.716%     | 0.256%     | 0.060%   |
| 19 | STL    |         1492 |             -3 |              -1 | 0.008%     | <0.012%        | 0.008%           | 0.008%     | <0.012%    | <0.012%  |
| 20 | FLA    |         1492 |             -7 |              -5 | 0.004%     | <0.012%        | 0.004%           | <0.012%    | <0.012%    | <0.012%  |
| 21 | TEX    |         1483 |              1 |               2 | 21.516%    | 15.804%        | 21.516%          | 4.636%     | 1.288%     | 0.256%   |
| 22 | SEA    |         1482 |              5 |              -2 | 3.284%     | 2.856%         | 3.284%           | 0.684%     | 0.176%     | 0.044%   |
| 23 | KCR    |         1478 |             -1 |              19 | <0.012%    | <0.012%        | <0.012%          | <0.012%    | <0.012%    | <0.012%  |
| 24 | WSN    |         1473 |             -2 |             -14 | <0.012%    | <0.012%        | <0.012%          | <0.012%    | <0.012%    | <0.012%  |
| 25 | SFG    |         1471 |              2 |              -1 | <0.012%    | <0.012%        | <0.012%          | <0.012%    | <0.012%    | <0.012%  |
| 26 | MIN    |         1471 |             -3 |              -7 | 0.900%     | 0.156%         | 0.900%           | 0.132%     | 0.044%     | 0.012%   |
| 27 | CIN    |         1463 |            -12 |             -17 | <0.012%    | <0.012%        | <0.012%          | <0.012%    | <0.012%    | <0.012%  |
| 28 | ANA    |         1452 |              3 |               5 | <0.012%    | <0.012%        | <0.012%          | <0.012%    | <0.012%    | <0.012%  |
| 29 | OAK    |         1428 |             -2 |              -1 | <0.012%    | <0.012%        | <0.012%          | <0.012%    | <0.012%    | <0.012%  |
| 30 | COL    |         1418 |             -8 |             -15 | <0.012%    | <0.012%        | <0.012%          | <0.012%    | <0.012%    | <0.012%  |