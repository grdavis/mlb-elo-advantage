# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-27 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 7% of games and have a 21.0% ROI over the last 7 days. ROI is -41.5% over the last 30 days and 1.52% over the last 365.

| Date       | Away   | Home   | Away Pitcher       | Home Pitcher      |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:-------------------|:------------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-27 | BAL    | NYY    | Shane Baz          | Elmer Rodríguez   |       40.98 |       59.02 | NA        | n/a              | NA        | -117             |
| 2026-09-27 | NYM    | WSN    | Sean Manaea        | DJ Herz           |       50.45 |       49.55 | NA        | +120             | NA        | +124             |
| 2026-09-27 | HOU    | OAK    | Peter Lambert      | Seth Johnson      |       57.93 |       42.07 | -201      | -112             | 164       | n/a              |
| 2026-09-27 | CHC    | BOS    | Shota Imanaga      | Tanner Houck      |       52.17 |       47.83 | -117      | +112             | -103      | +133             |
| 2026-09-27 | TBD    | PHI    | Nick Martinez      | Zack Wheeler      |       49.15 |       50.85 | 169       | +127             | -207      | +118             |
| 2026-09-27 | LAD    | SFG    | Jack Dreyer        | Carson Seymour    |       63.29 |       36.71 | -249      | -140             | 201       | n/a              |
| 2026-09-27 | CIN    | TOR    | Brandon Williamson | Max Scherzer      |       42.62 |       57.38 | 141       | n/a              | -171      | -110             |
| 2026-09-27 | COL    | CHW    | Kyle Freeland      | Anthony Kay       |       33.01 |       66.99 | 135       | n/a              | -163      | -163             |
| 2026-09-27 | PIT    | DET    | Jared Jones        | River Ryan        |       47.5  |       52.5  | -104      | +135             | -115      | +111             |
| 2026-09-27 | CLE    | KCR    | Parker Messick     | Daniel Lynch IV   |       57.97 |       42.03 | -109      | -113             | -111      | n/a              |
| 2026-09-27 | ATL    | FLA    | JR Ritchie         | Janson Junk       |       50.57 |       49.43 | 105       | +119             | -126      | +125             |
| 2026-09-27 | STL    | MIL    | Andre Pallante     | Jacob Misiorowski |       32.12 |       67.88 | 178       | n/a              | -218      | -169             |
| 2026-09-27 | TEX    | MIN    | MacKenzie Gore     | Dean Kremer       |       48.03 |       51.97 | -117      | +132             | -103      | +113             |
| 2026-09-27 | ARI    | SDP    | Michael Soroka     | Randy Vásquez     |       44.45 |       55.55 | -158      | +153             | 130       | -102             |
| 2026-09-27 | ANA    | SEA    | Yusei Kikuchi      | Logan Gilbert     |       43.97 |       56.03 | 137       | +157             | -166      | -104             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1578 |              5 |              11 | 100.000%   | 100.000%       | 100.000%         | 66.964%    | 41.330%    | 29.012%  |
|  2 | LAD    |         1565 |              1 |              13 | 100.000%   | 100.000%       | 100.000%         | 66.088%    | 32.086%    | 20.792%  |
|  3 | NYY    |         1546 |              0 |               8 | 100.000%   | <0.006%        | 100.000%         | 28.992%    | 19.548%    | 8.432%   |
|  4 | TBD    |         1542 |              4 |              19 | 100.000%   | 100.000%       | 100.000%         | 51.286%    | 33.940%    | 13.736%  |
|  5 | CHC    |         1542 |             -9 |              -3 | 100.000%   | <0.006%        | 100.000%         | 16.170%    | 7.492%     | 4.014%   |
|  6 | SDP    |         1537 |              0 |              12 | 100.000%   | <0.006%        | 100.000%         | 17.808%    | 7.926%     | 4.140%   |
|  7 | BOS    |         1537 |              1 |             -14 | 100.000%   | <0.006%        | 100.000%         | 19.722%    | 12.630%    | 4.936%   |
|  8 | ATL    |         1531 |             -4 |              -2 | 100.000%   | 100.000%       | 100.000%         | 20.280%    | 7.094%     | 3.492%   |
|  9 | PHI    |         1522 |            -10 |             -10 | 88.972%    | <0.006%        | 88.972%          | 11.434%    | 3.702%     | 1.746%   |
| 10 | ARI    |         1515 |              8 |               4 | 11.028%    | <0.006%        | 11.028%          | 1.256%     | 0.370%     | 0.166%   |
| 11 | CLE    |         1512 |              7 |              11 | 100.000%   | 100.000%       | 100.000%         | 57.112%    | 20.828%    | 6.212%   |
| 12 | DET    |         1509 |              2 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 13 | PIT    |         1506 |             -3 |               9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 14 | CHW    |         1504 |              3 |              -6 | 100.000%   | <0.006%        | 100.000%         | 22.116%    | 7.294%     | 2.082%   |
| 15 | NYM    |         1503 |              6 |              14 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 16 | BAL    |         1502 |              8 |               3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | TOR    |         1499 |             -7 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 18 | FLA    |         1495 |              4 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 19 | HOU    |         1493 |              7 |              -1 | 55.250%    | 55.250%        | 55.250%          | 12.046%    | 3.572%     | 0.748%   |
| 20 | TEX    |         1486 |             -5 |               7 | 44.750%    | 44.750%        | 44.750%          | 8.726%     | 2.188%     | 0.492%   |
| 21 | STL    |         1481 |             -2 |             -13 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | WSN    |         1479 |             -1 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | SEA    |         1475 |             -5 |              -7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 24 | MIN    |         1470 |              2 |               2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | CIN    |         1467 |              7 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | SFG    |         1465 |             -3 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | KCR    |         1461 |            -10 |             -22 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1445 |             -3 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1422 |              1 |              -5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1410 |             -4 |             -15 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |