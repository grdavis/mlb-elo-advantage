# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-02 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 9% of games and have a 29.37% ROI over the last 7 days. ROI is -22.84% over the last 30 days and 1.93% over the last 365.

| Date   | Away   | Home   | Away Pitcher   | Home Pitcher   | Away WinP   | Home WinP   | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|--------|--------|--------|----------------|----------------|-------------|-------------|-----------|------------------|-----------|------------------|

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when that outcome is still possible but did not occur in any simulation. A shown 0.00% means the team has no remaining path to that outcome.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1579 |              2 |              14 | 100.000%   | 100.000%       | 100.000%         | 62.902%    | 38.714%    | 26.776%  |
|  2 | LAD    |         1566 |              3 |              19 | 100.000%   | 100.000%       | 100.000%         | 63.858%    | 31.792%    | 20.452%  |
|  3 | NYY    |         1553 |              7 |              11 | 100.000%   | 0.00%          | 100.000%         | 53.642%    | 35.904%    | 16.134%  |
|  4 | SDP    |         1546 |             11 |              26 | 100.000%   | 0.00%          | 100.000%         | 37.098%    | 16.778%    | 8.914%   |
|  5 | TBD    |         1539 |              2 |              17 | 100.000%   | 100.000%       | 100.000%         | 46.358%    | 29.064%    | 11.530%  |
|  6 | CHC    |         1539 |             -3 |              -7 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  7 | ATL    |         1532 |              4 |               0 | 100.000%   | 100.000%       | 100.000%         | 36.142%    | 12.716%    | 6.246%   |
|  8 | BOS    |         1527 |            -10 |             -12 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  9 | PHI    |         1521 |             -6 |             -18 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 10 | ARI    |         1512 |             -5 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 11 | CHW    |         1511 |              3 |               4 | 100.000%   | 0.00%          | 100.000%         | 48.692%    | 16.674%    | 4.730%   |
| 12 | CLE    |         1511 |              2 |              12 | 100.000%   | 100.000%       | 100.000%         | 51.308%    | 18.358%    | 5.218%   |
| 13 | PIT    |         1508 |              0 |               7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 14 | DET    |         1507 |              0 |               6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 15 | BAL    |         1502 |              0 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 16 | TOR    |         1501 |             -1 |              -7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 17 | NYM    |         1501 |              1 |               8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 18 | FLA    |         1497 |             -1 |              -5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 19 | HOU    |         1491 |              2 |              -4 | 100.000%   | 100.000%       | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 20 | TEX    |         1484 |              1 |               4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 21 | WSN    |         1481 |             -2 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 22 | STL    |         1480 |             -2 |             -16 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 23 | SEA    |         1477 |              4 |              -7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 24 | MIN    |         1472 |             -1 |              -5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 25 | CIN    |         1464 |              1 |              -7 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 26 | SFG    |         1463 |             -3 |              -9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 27 | KCR    |         1463 |             -1 |             -15 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 28 | ANA    |         1443 |             -5 |              -4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 29 | OAK    |         1418 |             -8 |              -6 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 30 | COL    |         1409 |              2 |             -17 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |