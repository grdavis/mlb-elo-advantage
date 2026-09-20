# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-20 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 4% of games and have a 14.25% ROI over the last 7 days. ROI is -33.95% over the last 30 days and 0.95% over the last 365.

| Date       | Away   | Home   | Away Pitcher       | Home Pitcher    |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:-------------------|:----------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-20 | PHI    | NYM    | Cristopher Sánchez | Jonah Tong      |       50.62 |       49.38 | NA        | +119             | NA        | +125             |
| 2026-09-20 | KCR    | PIT    | Michael Wacha      | Lake Bachar     |       41.38 |       58.62 | 100       | n/a              | -120      | -116             |
| 2026-09-20 | CHC    | CIN    | David Peterson     | Rhett Lowder    |       61.2  |       38.8  | -163      | -128             | 134       | n/a              |
| 2026-09-20 | OAK    | CLE    | Jack Perkins       | Gavin Williams  |       33.83 |       66.17 | 201       | n/a              | -249      | -158             |
| 2026-09-20 | BOS    | TBD    | Patrick Sandoval   | Griffin Jax     |       45.35 |       54.65 | 127       | +148             | -153      | +101             |
| 2026-09-20 | DET    | CHW    | Troy Melton        | Davis Martin    |       49.25 |       50.75 | -102      | +126             | -118      | +119             |
| 2026-09-20 | ATL    | HOU    | Martín Pérez       | Hunter Brown    |       52.88 |       47.12 | 130       | +109             | -157      | +137             |
| 2026-09-20 | WSN    | STL    | Jake Irvin         | Quinn Mathews   |       45.34 |       54.66 | 108       | +148             | -131      | +101             |
| 2026-09-20 | TOR    | TEX    | Spencer Miles      | Jacob deGrom    |       46.6  |       53.4  | 149       | +140             | -181      | +107             |
| 2026-09-20 | SEA    | COL    | Kade Anderson      | Tomoyuki Sugano |       58.24 |       41.76 | -163      | -114             | 134       | n/a              |
| 2026-09-20 | MIN    | ANA    | Dean Kremer        | Ryan Johnson    |       47.63 |       52.37 | -117      | +135             | -103      | +111             |
| 2026-09-20 | NYY    | ARI    | Will Warren        | Corbin Burnes   |       53.61 |       46.39 | -101      | +106             | -119      | +142             |
| 2026-09-20 | SFG    | LAD    | Matt Wilkinson     |                 |       31.82 |       68.18 | 230       | n/a              | -288      | -172             |
| 2026-09-20 | FLA    | SDP    | Sandy Alcantara    | Walker Buehler  |       39.93 |       60.07 | 129       | n/a              | -156      | -123             |
| 2026-09-20 | MIL    | BAL    | Jacob Misiorowski  | Brandon Young   |       61.18 |       38.82 | -206      | -128             | 169       | n/a              |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 47169 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1571 |              3 |              10 | 100.000%   | 100.000%       | 100.000%         | 63.752%    | 37.444%    | 25.347%  |
|  2 | LAD    |         1563 |              7 |              11 | 100.000%   | 100.000%       | 100.000%         | 62.774%    | 31.807%    | 20.679%  |
|  3 | NYY    |         1549 |              5 |              10 | 100.000%   | 3.888%         | 100.000%         | 32.886%    | 23.217%    | 10.575%  |
|  4 | CHC    |         1548 |              2 |               4 | 99.843%    | <0.006%        | 99.843%          | 21.626%    | 10.551%    | 6.178%   |
|  5 | BOS    |         1539 |             -3 |             -11 | 100.000%   | <0.006%        | 100.000%         | 20.541%    | 13.990%    | 5.807%   |
|  6 | TBD    |         1536 |              4 |              19 | 100.000%   | 96.112%        | 100.000%         | 46.641%    | 30.900%    | 12.506%  |
|  7 | SDP    |         1535 |              5 |              10 | 98.982%    | <0.006%        | 98.982%          | 16.384%    | 6.871%     | 3.685%   |
|  8 | ATL    |         1533 |              3 |               5 | 100.000%   | 99.994%        | 100.000%         | 20.420%    | 7.683%     | 4.092%   |
|  9 | PHI    |         1529 |             -5 |               0 | 98.287%    | 0.006%         | 98.287%          | 14.766%    | 5.588%     | 2.828%   |
| 10 | DET    |         1511 |              2 |              -8 | 0.006%     | <0.006%        | 0.006%           | <0.006%    | <0.006%    | <0.006%  |
| 11 | PIT    |         1508 |              2 |               7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 12 | ARI    |         1504 |             -4 |             -10 | 2.887%     | <0.006%        | 2.887%           | 0.278%     | 0.055%     | 0.023%   |
| 13 | CLE    |         1504 |              4 |               9 | 95.143%    | 59.804%        | 95.143%          | 40.158%    | 13.693%    | 3.755%   |
| 14 | TOR    |         1503 |            -10 |              -2 | 5.217%     | <0.006%        | 5.217%           | 1.253%     | 0.441%     | 0.112%   |
| 15 | NYM    |         1500 |             -3 |               7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 16 | CHW    |         1498 |              1 |             -12 | 90.506%    | 40.196%        | 90.506%          | 30.200%    | 9.498%     | 2.474%   |
| 17 | BAL    |         1496 |              5 |               3 | 0.469%     | <0.006%        | 0.469%           | 0.087%     | 0.019%     | 0.008%   |
| 18 | TEX    |         1494 |             10 |              12 | 72.715%    | 69.575%        | 72.715%          | 19.954%    | 5.900%     | 1.384%   |
| 19 | FLA    |         1493 |             -1 |              -6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 20 | HOU    |         1488 |             -3 |              -4 | 35.920%    | 30.403%        | 35.920%          | 8.272%     | 2.343%     | 0.547%   |
| 21 | WSN    |         1482 |              8 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | STL    |         1481 |            -12 |             -21 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | SEA    |         1479 |             -1 |              -7 | 0.023%     | 0.021%         | 0.023%           | 0.008%     | <0.006%    | <0.006%  |
| 24 | KCR    |         1472 |             -5 |               3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | SFG    |         1470 |              1 |               5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | MIN    |         1463 |             -4 |             -14 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | CIN    |         1463 |             -2 |              -5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1452 |              1 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1422 |             -8 |               1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1415 |             -1 |             -11 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |