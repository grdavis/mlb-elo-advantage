# MLB Elo Game Predictions and Playoff Probabilities for 2026-10-15 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 29% of games and have a 130.5% ROI over the last 7 days. ROI is 20.6% over the last 30 days and 1.52% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher   |   Away WinP |   Home WinP | Away ML   |   Away Threshold | Home ML   |   Home Threshold |
|:-----------|:-------|:-------|:---------------|:---------------|------------:|------------:|:----------|-----------------:|:----------|-----------------:|
| 2026-10-15 | TBD    | CLE    | Griffin Jax    | Gavin Williams |       51.84 |       48.16 | NA        |             +114 | NA        |             +132 |
| 2026-10-15 | MIL    | LAD    | Shane Drohan   | Tyler Glasnow  |       47.61 |       52.39 | 130       |             +135 | -157      |             +111 |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when that outcome is still possible but did not occur in any simulation. A shown 0.00% means the team has no remaining path to that outcome.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1591 |              9 |              21 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 96.164%    | 70.298%  |
|  2 | LAD    |         1563 |             -9 |               3 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 3.836%     | 2.470%   |
|  3 | NYY    |         1546 |              0 |              -3 | 100.000%   | 0.00%          | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
|  4 | TBD    |         1544 |             -3 |              11 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 57.366%    | 17.558%  |
|  5 | SDP    |         1542 |              0 |              16 | 100.000%   | 0.00%          | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
|  6 | CHC    |         1539 |              0 |              -7 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  7 | ATL    |         1527 |              0 |              -4 | 100.000%   | 100.000%       | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
|  8 | BOS    |         1527 |              0 |             -13 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
|  9 | PHI    |         1521 |              0 |             -10 | 100.000%   | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 10 | CLE    |         1520 |              5 |              22 | 100.000%   | 100.000%       | 100.000%         | 100.000%   | 42.634%    | 9.674%   |
| 11 | ARI    |         1512 |              0 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 12 | PIT    |         1508 |              0 |               4 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 13 | DET    |         1507 |              0 |              -8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 14 | CHW    |         1504 |             -2 |               6 | 100.000%   | 0.00%          | 100.000%         | 0.00%      | 0.00%      | 0.00%    |
| 15 | BAL    |         1502 |              0 |               8 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 16 | TOR    |         1501 |              0 |              -5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 17 | NYM    |         1501 |              0 |               1 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 18 | FLA    |         1497 |              0 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 19 | HOU    |         1491 |              0 |              -2 | 100.000%   | 100.000%       | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 20 | TEX    |         1484 |              0 |              -3 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 21 | WSN    |         1481 |              0 |               5 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 22 | STL    |         1480 |              0 |             -10 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 23 | SEA    |         1477 |              0 |               0 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 24 | MIN    |         1472 |              0 |              10 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 25 | CIN    |         1464 |              0 |               2 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 26 | SFG    |         1463 |              0 |              -9 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 27 | KCR    |         1463 |              0 |             -12 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 28 | ANA    |         1443 |              0 |             -12 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 29 | OAK    |         1418 |              0 |             -11 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |
| 30 | COL    |         1409 |              0 |             -10 | 0.00%      | 0.00%          | 0.00%            | 0.00%      | 0.00%      | 0.00%    |