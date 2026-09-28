# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-28 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 7% of games and have a 10.12% ROI over the last 7 days. ROI is -30.62% over the last 30 days and 1.95% over the last 365.

| Date   | Away   | Home   | Away Pitcher   | Home Pitcher   | Away WinP   | Home WinP   | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|--------|--------|--------|----------------|----------------|-------------|-------------|-----------|------------------|-----------|------------------|

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1579 |              6 |              11 | 100.000%   | 100.000%       | 100.000%         | 66.350%    | 40.854%    | 28.842%  |
|  2 | LAD    |         1566 |              2 |              15 | 100.000%   | 100.000%       | 100.000%         | 66.944%    | 32.472%    | 21.550%  |
|  3 | NYY    |         1546 |              0 |               8 | 100.000%   | <0.006%        | 100.000%         | 30.886%    | 20.710%    | 8.638%   |
|  4 | CHC    |         1545 |             -6 |              -4 | 100.000%   | <0.006%        | 100.000%         | 15.918%    | 7.704%     | 4.400%   |
|  5 | SDP    |         1540 |              3 |              17 | 100.000%   | <0.006%        | 100.000%         | 17.732%    | 8.138%     | 4.512%   |
|  6 | TBD    |         1539 |              1 |              14 | 100.000%   | 100.000%       | 100.000%         | 50.036%    | 32.334%    | 12.586%  |
|  7 | BOS    |         1534 |             -2 |             -17 | 100.000%   | <0.006%        | 100.000%         | 19.078%    | 12.010%    | 4.540%   |
|  8 | ATL    |         1529 |             -6 |              -5 | 100.000%   | 100.000%       | 100.000%         | 19.278%    | 6.292%     | 3.162%   |
|  9 | PHI    |         1525 |             -7 |              -9 | 100.000%   | <0.006%        | 100.000%         | 13.778%    | 4.540%     | 2.236%   |
| 10 | ARI    |         1512 |              5 |               1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 11 | CLE    |         1511 |              6 |              14 | 100.000%   | 100.000%       | 100.000%         | 55.172%    | 20.156%    | 5.786%   |
| 12 | PIT    |         1508 |             -1 |               8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 13 | DET    |         1507 |             -4 |              -5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 14 | CHW    |         1505 |              4 |              -6 | 100.000%   | <0.006%        | 100.000%         | 22.084%    | 7.606%     | 1.980%   |
| 15 | BAL    |         1502 |              7 |               1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 16 | TOR    |         1501 |             -4 |              -1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | NYM    |         1501 |              4 |               9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 18 | HOU    |         1497 |             11 |               6 | 100.000%   | 100.000%       | 100.000%         | 22.744%    | 7.184%     | 1.768%   |
| 19 | FLA    |         1497 |              6 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 20 | TEX    |         1484 |             -7 |               6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 21 | WSN    |         1481 |              4 |               1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | STL    |         1480 |             -3 |             -11 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | SEA    |         1477 |             -3 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 24 | MIN    |         1472 |              6 |               5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | CIN    |         1464 |              4 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | SFG    |         1463 |             -8 |              -5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | KCR    |         1463 |             -8 |             -24 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1443 |             -5 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1418 |             -3 |              -7 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1409 |             -5 |             -15 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |