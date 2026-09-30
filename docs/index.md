# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-30 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 8% of games and have a 10.12% ROI over the last 7 days. ROI is -35.86% over the last 30 days and 1.47% over the last 365.

| Date       | Away   | Home   | Away Pitcher       | Home Pitcher   |   Away WinP |   Home WinP | Away ML   |   Away Threshold | Home ML   |   Home Threshold |
|:-----------|:-------|:-------|:-------------------|:---------------|------------:|------------:|:----------|-----------------:|:----------|-----------------:|
| 2026-09-30 | PHI    | ATL    | Cristopher Sánchez | Tyler Mahle    |       45.66 |       54.34 | NA        |             +146 | NA        |             +103 |
| 2026-09-30 | CHW    | HOU    | Sean Burke         | Hunter Brown   |       47.52 |       52.48 | 129       |             +135 | -155      |             +111 |
| 2026-09-30 | BOS    | NYY    | Sonny Gray         | Max Fried      |       44.28 |       55.72 | 112       |             +155 | -136      |             -103 |
| 2026-09-30 | CHC    | SDP    | Kevin Gausman      | Nick Pivetta   |       44.58 |       55.42 | 123       |             +153 | -149      |             -102 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1579 |              6 |              20 | 100.000%   | 100.000%       | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
|  2 | LAD    |         1566 |              2 |              12 | 100.000%   | 100.000%       | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
|  3 | NYY    |         1550 |              2 |              11 | 100.000%   | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
|  4 | SDP    |         1543 |              6 |              18 | 100.000%   | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
|  5 | CHC    |         1541 |             -4 |             -11 | 100.000%   | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
|  6 | TBD    |         1539 |              3 |              15 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
|  7 | ATL    |         1531 |             -1 |               0 | <0.006%    | <0.006%        | 100.000%         | 44.208%    | <0.006%    | <0.006%  |
|  8 | BOS    |         1530 |             -6 |             -16 | 100.000%   | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
|  9 | PHI    |         1523 |             -9 |             -14 | 100.000%   | 100.000%       | 100.000%         | 16.936%    | <0.006%    | <0.006%  |
| 10 | ARI    |         1512 |              2 |               3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 11 | CLE    |         1511 |              6 |              10 | 100.000%   | 100.000%       | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 12 | CHW    |         1508 |              6 |               2 | <0.006%    | <0.006%        | 100.000%         | 29.312%    | <0.006%    | <0.006%  |
| 13 | PIT    |         1508 |              0 |               6 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 14 | DET    |         1507 |              1 |               3 | 100.000%   | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 15 | BAL    |         1502 |              3 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 16 | TOR    |         1501 |              0 |              -5 | 100.000%   | 100.000%       | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | NYM    |         1501 |             -2 |              10 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 18 | FLA    |         1497 |              0 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 19 | HOU    |         1494 |              5 |              -2 | <0.006%    | <0.006%        | 100.000%         | 9.544%     | <0.006%    | <0.006%  |
| 20 | TEX    |         1484 |             -2 |               0 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 21 | WSN    |         1481 |              0 |               1 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | STL    |         1480 |             -5 |              -9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | SEA    |         1477 |              0 |               0 | 100.000%   | 100.000%       | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 24 | MIN    |         1472 |              3 |              -3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | CIN    |         1464 |              1 |              -3 | 100.000%   | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | SFG    |         1463 |             -4 |              -8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | KCR    |         1463 |             -7 |             -20 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1443 |              0 |              -8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1418 |             -8 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1409 |             -2 |             -13 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |