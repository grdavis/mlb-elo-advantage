# MLB Elo Game Predictions and Playoff Probabilities for 2026-07-13 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

The thresholds indicate at what odds the model thinks there is value in betting on a team. These thresholds were selected via backtesting over the last 2 years (2.7K games). For transparency, these recommendations have been triggered for 20% of games and have a -4.47% ROI over the last 7 days. ROI is -13.98% over the last 30 days and -2.53% over the last 365.

| Date   | Away   | Home   | Away WinP   | Home WinP   | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|--------|--------|--------|-------------|-------------|-----------|------------------|-----------|------------------|

# Team Elo Ratings
This table summarizes each team's Elo rating and their chances of making it to various stages of the postseason based on 12000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | LAD    |         1562 |            -14 |             -12 | 99.967%    | 99.708%        | 99.967%          | 64.892%    | 36.250%    | 24.225%  |
|  2 | MIL    |         1561 |             -5 |               1 | 99.858%    | 92.033%        | 99.858%          | 62.058%    | 34.683%    | 23.842%  |
|  3 | NYY    |         1535 |              5 |             -22 | 97.658%    | 42.133%        | 97.658%          | 44.275%    | 27.683%    | 12.358%  |
|  4 | CHC    |         1532 |              4 |              17 | 83.208%    | 7.200%         | 83.208%          | 18.800%    | 8.375%     | 4.725%   |
|  5 | BOS    |         1528 |             15 |              17 | 57.750%    | 2.850%         | 57.750%          | 16.150%    | 9.333%     | 3.808%   |
|  6 | ATL    |         1525 |             -1 |             -23 | 91.092%    | 59.308%        | 91.092%          | 20.575%    | 8.500%     | 4.658%   |
|  7 | PHI    |         1524 |              1 |              -1 | 79.117%    | 31.233%        | 79.117%          | 15.717%    | 6.342%     | 3.325%   |
|  8 | TBD    |         1515 |              2 |              13 | 98.417%    | 54.625%        | 98.417%          | 35.767%    | 19.200%    | 6.933%   |
|  9 | CLE    |         1511 |              6 |               0 | 82.683%    | 58.633%        | 82.683%          | 33.233%    | 14.783%    | 5.025%   |
| 10 | SEA    |         1510 |             -7 |             -11 | 54.042%    | 40.258%        | 54.042%          | 17.425%    | 7.825%     | 2.467%   |
| 11 | PIT    |         1510 |              7 |              15 | 23.317%    | 0.308%         | 23.317%          | 3.192%     | 1.067%     | 0.467%   |
| 12 | FLA    |         1509 |             -1 |              16 | 45.042%    | 8.750%         | 45.042%          | 6.258%     | 2.150%     | 0.983%   |
| 13 | DET    |         1505 |              5 |              21 | 14.617%    | 3.933%         | 14.617%          | 3.583%     | 1.467%     | 0.442%   |
| 14 | TOR    |         1503 |              7 |             -10 | 15.058%    | 0.250%         | 15.058%          | 3.058%     | 1.258%     | 0.375%   |
| 15 | ARI    |         1503 |             10 |               9 | 25.000%    | 0.158%         | 25.000%          | 2.767%     | 0.883%     | 0.375%   |
| 16 | STL    |         1501 |             -1 |              -2 | 31.333%    | 0.458%         | 31.333%          | 3.667%     | 1.125%     | 0.417%   |
| 17 | CHW    |         1500 |              0 |               9 | 64.100%    | 31.233%        | 64.100%          | 18.800%    | 7.733%     | 2.308%   |
| 18 | TEX    |         1498 |             -3 |              -7 | 58.575%    | 45.767%        | 58.575%          | 16.417%    | 6.583%     | 1.850%   |
| 19 | SDP    |         1498 |              4 |              -8 | 14.333%    | 0.133%         | 14.333%          | 1.583%     | 0.500%     | 0.225%   |
| 20 | BAL    |         1494 |              5 |               5 | 11.450%    | 0.142%         | 11.450%          | 2.150%     | 0.833%     | 0.267%   |
| 21 | HOU    |         1488 |             -1 |              -1 | 23.033%    | 13.800%        | 23.033%          | 4.992%     | 1.758%     | 0.492%   |
| 22 | MIN    |         1484 |              2 |              25 | 22.267%    | 6.200%         | 22.267%          | 4.125%     | 1.525%     | 0.383%   |
| 23 | WSN    |         1483 |             -4 |               2 | 7.375%     | 0.708%         | 7.375%           | 0.467%     | 0.125%     | 0.050%   |
| 24 | CIN    |         1477 |             -1 |               1 | 0.317%     | <0.025%        | 0.317%           | 0.017%     | <0.025%    | <0.025%  |
| 25 | SFG    |         1476 |             -4 |               1 | 0.042%     | <0.025%        | 0.042%           | 0.008%     | <0.025%    | <0.025%  |
| 26 | NYM    |         1470 |             -4 |             -19 | <0.025%    | <0.025%        | <0.025%          | <0.025%    | <0.025%    | <0.025%  |
| 27 | KCR    |         1464 |             -9 |             -11 | 0.025%     | <0.025%        | 0.025%           | <0.025%    | <0.025%    | <0.025%  |
| 28 | ANA    |         1455 |              2 |              -9 | 0.033%     | 0.017%         | 0.033%           | <0.025%    | <0.025%    | <0.025%  |
| 29 | OAK    |         1449 |            -16 |             -37 | 0.292%     | 0.158%         | 0.292%           | 0.025%     | 0.017%     | <0.025%  |
| 30 | COL    |         1431 |             -2 |              20 | <0.025%    | <0.025%        | <0.025%          | <0.025%    | <0.025%    | <0.025%  |