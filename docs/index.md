# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-17 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 6% of games and have a -51.2% ROI over the last 7 days. ROI is -41.04% over the last 30 days and -1.42% over the last 365.

| Date       | Away   | Home   | Away Pitcher     | Home Pitcher    |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:-----------------|:----------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-17 | MIL    | PIT    | Kyle Harrison    | Wilber Dotel    |       53.69 |       46.31 | NA        | +105             | NA        | +142             |
| 2026-09-17 | LAD    | CIN    | Justin Wrobleski | Brady Singer    |       61.39 |       38.61 | NA        | -129             | NA        | n/a              |
| 2026-09-17 | OAK    | TBD    | Jeffrey Springs  | Drew Rasmussen  |       29.69 |       70.31 | NA        | n/a              | NA        | -188             |
| 2026-09-17 | SDP    | COL    | Michael King     | Tanner Gordon   |       65.47 |       34.53 | -203      | -153             | 166       | n/a              |
| 2026-09-17 | KCR    | HOU    | Seth Lugo        |                 |       42.66 |       57.34 | 128       | n/a              | -155      | -110             |
| 2026-09-17 | PHI    | NYM    | Aaron Nola       | Nolan McLean    |       52.81 |       47.19 | 109       | +109             | -132      | +137             |
| 2026-09-17 | DET    | CHW    | Framber Valdez   | Erick Fedde     |       49.73 |       50.27 | -103      | +124             | -117      | +121             |
| 2026-09-17 | BOS    | TEX    | Sonny Gray       | Tyler Alexander |       54.19 |       45.81 | -123      | +103             | 102       | +145             |
| 2026-09-17 | MIN    | ANA    | Taj Bradley      | Walbert Ureña   |       48.66 |       51.34 | -112      | +129             | -108      | +116             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 34482 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1571 |              7 |              10 | 100.000%   | 100.000%       | 100.000%         | 64.770%    | 39.461%    | 27.426%  |
|  2 | LAD    |         1556 |             -2 |               6 | 100.000%   | 100.000%       | 100.000%         | 60.956%    | 29.024%    | 18.015%  |
|  3 | CHC    |         1548 |              5 |               0 | 99.751%    | <0.009%        | 99.751%          | 22.209%    | 11.270%    | 6.748%   |
|  4 | NYY    |         1547 |              0 |              14 | 100.000%   | 7.346%         | 100.000%         | 34.061%    | 24.010%    | 10.867%  |
|  5 | BOS    |         1537 |             -3 |             -13 | 99.962%    | <0.009%        | 99.962%          | 20.698%    | 13.828%    | 5.818%   |
|  6 | TBD    |         1534 |              8 |              13 | 100.000%   | 92.654%        | 100.000%         | 45.432%    | 30.204%    | 12.404%  |
|  7 | PHI    |         1533 |             -4 |               7 | 98.689%    | 1.395%         | 98.689%          | 16.330%    | 7.152%     | 3.764%   |
|  8 | ATL    |         1528 |              0 |              -1 | 100.000%   | 98.605%        | 100.000%         | 20.788%    | 7.294%     | 3.651%   |
|  9 | SDP    |         1528 |              3 |               3 | 93.927%    | <0.009%        | 93.927%          | 14.193%    | 5.565%     | 2.909%   |
| 10 | DET    |         1512 |              9 |             -11 | 0.737%     | 0.244%         | 0.737%           | 0.247%     | 0.131%     | 0.029%   |
| 11 | TOR    |         1509 |             -1 |               4 | 25.805%    | <0.009%        | 25.805%          | 6.867%     | 2.482%     | 0.687%   |
| 12 | ARI    |         1505 |             -4 |              -4 | 7.618%     | <0.009%        | 7.618%           | 0.751%     | 0.232%     | 0.078%   |
| 13 | PIT    |         1502 |             -7 |               1 | 0.006%     | <0.009%        | 0.006%           | 0.003%     | 0.003%     | <0.009%  |
| 14 | CLE    |         1500 |              5 |               9 | 80.178%    | 53.950%        | 80.178%          | 31.892%    | 10.869%    | 2.828%   |
| 15 | BAL    |         1498 |              4 |               2 | 4.742%     | <0.009%        | 4.742%           | 1.056%     | 0.351%     | 0.070%   |
| 16 | FLA    |         1497 |              4 |              -3 | 0.009%     | <0.009%        | 0.009%           | <0.009%    | <0.009%    | <0.009%  |
| 17 | NYM    |         1496 |             -5 |               3 | <0.009%    | <0.009%        | <0.009%          | <0.009%    | <0.009%    | <0.009%  |
| 18 | CHW    |         1496 |             -4 |             -11 | 74.694%    | 45.798%        | 74.694%          | 27.652%    | 8.761%     | 2.317%   |
| 19 | TEX    |         1490 |              7 |               7 | 54.251%    | 47.471%        | 54.251%          | 15.425%    | 4.533%     | 1.215%   |
| 20 | HOU    |         1490 |             -8 |              -7 | 59.457%    | 52.393%        | 59.457%          | 16.638%    | 4.820%     | 1.169%   |
| 21 | STL    |         1489 |             -1 |             -14 | <0.009%    | <0.009%        | <0.009%          | <0.009%    | <0.009%    | <0.009%  |
| 22 | SEA    |         1480 |              3 |              -4 | 0.154%     | 0.136%         | 0.154%           | 0.026%     | 0.009%     | 0.006%   |
| 23 | KCR    |         1478 |              0 |              15 | <0.009%    | <0.009%        | <0.009%          | <0.009%    | <0.009%    | <0.009%  |
| 24 | SFG    |         1474 |              0 |               7 | <0.009%    | <0.009%        | <0.009%          | <0.009%    | <0.009%    | <0.009%  |
| 25 | WSN    |         1474 |              4 |              -7 | <0.009%    | <0.009%        | <0.009%          | <0.009%    | <0.009%    | <0.009%  |
| 26 | CIN    |         1465 |             -4 |              -6 | <0.009%    | <0.009%        | <0.009%          | <0.009%    | <0.009%    | <0.009%  |
| 27 | MIN    |         1464 |             -8 |             -13 | 0.020%     | 0.009%         | 0.020%           | 0.006%     | 0.003%     | <0.009%  |
| 28 | ANA    |         1452 |             -3 |               4 | <0.009%    | <0.009%        | <0.009%          | <0.009%    | <0.009%    | <0.009%  |
| 29 | OAK    |         1428 |             -5 |               1 | <0.009%    | <0.009%        | <0.009%          | <0.009%    | <0.009%    | <0.009%  |
| 30 | COL    |         1417 |             -4 |             -14 | <0.009%    | <0.009%        | <0.009%          | <0.009%    | <0.009%    | <0.009%  |