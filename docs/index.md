# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-22 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 8% of games and have a 33.71% ROI over the last 7 days. ROI is -25.93% over the last 30 days and 1.72% over the last 365.

| Date       | Away   | Home   | Away Pitcher       | Home Pitcher    |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:-------------------|:----------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-22 | TBD    | NYY    | Drew Rasmussen     | Ryan Weathers   |       47.07 |       52.93 | NA        | +138             | NA        | +109             |
| 2026-09-22 | TOR    | BAL    |                    |                 |       48.47 |       51.53 | 102       | +130             | -123      | +115             |
| 2026-09-22 | WSN    | DET    | Riley Cornelio     | Drew Anderson   |       43.29 |       56.71 | 130       | +161             | -157      | -107             |
| 2026-09-22 | MIL    | PHI    | Dustin May         | Zack Wheeler    |       51.52 |       48.48 | 118       | +115             | -142      | +130             |
| 2026-09-22 | STL    | PIT    | Andre Pallante     | Jared Jones     |       41.87 |       58.13 | 143       | n/a              | -173      | -113             |
| 2026-09-22 | CLE    | BOS    | Parker Messick     | Payton Tolle    |       41.94 |       58.06 | 108       | n/a              | -131      | -113             |
| 2026-09-22 | TBD    | NYY    | Nick Martinez      | Carlos Rodón    |       45.11 |       54.89 | 112       | +149             | -135      | +100             |
| 2026-09-22 | CIN    | ATL    | Brandon Williamson | JR Ritchie      |       37.29 |       62.71 | 158       | n/a              | -193      | -136             |
| 2026-09-22 | FLA    | CHC    | Janson Junk        | Shota Imanaga   |       37.37 |       62.63 | 169       | n/a              | -207      | -136             |
| 2026-09-22 | CHW    | KCR    | Anthony Kay        | Daniel Lynch IV |       51.02 |       48.98 | -125      | +117             | 104       | +127             |
| 2026-09-22 | NYM    | TEX    | Sean Manaea        | MacKenzie Gore  |       48.07 |       51.93 | 110       | +132             | -133      | +113             |
| 2026-09-22 | ARI    | COL    | Michael Soroka     | Kyle Freeland   |       62.66 |       37.34 | -198      | -136             | 162       | n/a              |
| 2026-09-22 | ANA    | OAK    | Yusei Kikuchi      | Brady Basso     |       51.09 |       48.91 | -108      | +117             | -111      | +128             |
| 2026-09-22 | HOU    | SEA    | Cristian Javier    | Logan Gilbert   |       48.42 |       51.58 | 113       | +130             | -137      | +115             |
| 2026-09-22 | MIN    | SFG    | Taj Bradley        | Anthony Molina  |       48.21 |       51.79 | -135      | +131             | 112       | +114             |
| 2026-09-22 | SDP    | LAD    | Michael King       |                 |       43.39 |       56.61 | 109       | +161             | -131      | -107             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1573 |              3 |              12 | 100.000%   | 100.000%       | 100.000%         | 63.756%    | 37.536%    | 25.792%  |
|  2 | LAD    |         1564 |              4 |               9 | 100.000%   | 100.000%       | 100.000%         | 62.664%    | 31.154%    | 20.384%  |
|  3 | CHC    |         1551 |              5 |               3 | 99.948%    | <0.006%        | 99.948%          | 21.896%    | 10.814%    | 6.420%   |
|  4 | NYY    |         1546 |             -3 |               6 | 100.000%   | 0.896%         | 100.000%         | 30.882%    | 21.406%    | 9.330%   |
|  5 | TBD    |         1538 |              5 |              20 | 100.000%   | 99.104%        | 100.000%         | 49.168%    | 32.800%    | 13.328%  |
|  6 | SDP    |         1537 |             11 |               8 | 99.110%    | <0.006%        | 99.110%          | 16.240%    | 7.032%     | 3.800%   |
|  7 | BOS    |         1536 |             -4 |             -16 | 100.000%   | <0.006%        | 100.000%         | 20.156%    | 13.200%    | 5.196%   |
|  8 | ATL    |         1535 |              4 |               7 | 100.000%   | 99.996%        | 100.000%         | 20.154%    | 7.572%     | 4.038%   |
|  9 | PHI    |         1532 |              1 |              -3 | 98.736%    | 0.004%         | 98.736%          | 15.074%    | 5.826%     | 3.056%   |
| 10 | DET    |         1511 |             -4 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 11 | PIT    |         1509 |              5 |              11 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 12 | ARI    |         1507 |              0 |              -5 | 2.206%     | <0.006%        | 2.206%           | 0.216%     | 0.066%     | 0.042%   |
| 13 | TOR    |         1505 |             -1 |               1 | 1.748%     | <0.006%        | 1.748%           | 0.390%     | 0.136%     | 0.038%   |
| 14 | CLE    |         1505 |              7 |               6 | 98.786%    | 55.996%        | 98.786%          | 41.186%    | 14.102%    | 3.926%   |
| 15 | CHW    |         1501 |              3 |              -6 | 97.668%    | 44.004%        | 97.668%          | 35.072%    | 11.776%    | 3.166%   |
| 16 | NYM    |         1497 |             -3 |               1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | BAL    |         1495 |              1 |               3 | 0.074%     | <0.006%        | 0.074%           | 0.012%     | 0.002%     | <0.006%  |
| 18 | TEX    |         1491 |              4 |              10 | 68.676%    | 67.926%        | 68.676%          | 16.272%    | 4.690%     | 1.064%   |
| 19 | FLA    |         1491 |             -4 |             -12 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 20 | HOU    |         1486 |             -7 |              -2 | 32.984%    | 32.010%        | 32.984%          | 6.850%     | 1.882%     | 0.418%   |
| 21 | STL    |         1483 |             -7 |             -14 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | SEA    |         1480 |              3 |              -2 | 0.064%     | 0.064%         | 0.064%           | 0.012%     | 0.006%     | 0.002%   |
| 23 | WSN    |         1477 |              1 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 24 | SFG    |         1471 |             -1 |               8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | KCR    |         1471 |             -4 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | MIN    |         1466 |              4 |              -7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | CIN    |         1460 |             -2 |             -11 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1448 |             -7 |              -7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1421 |             -8 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1414 |             -5 |              -9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |