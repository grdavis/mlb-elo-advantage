# MLB Elo Game Predictions and Playoff Probabilities for 2026-07-14 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

The thresholds indicate at what odds the model thinks there is value in betting on a team. These thresholds were selected via backtesting over the last 2 years (2.7K games). For transparency, these recommendations have been triggered for 20% of games and have a -14.59% ROI over the last 7 days. ROI is -14.63% over the last 30 days and -2.53% over the last 365.

| Date   | Away   | Home   | Away WinP   | Home WinP   | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|--------|--------|--------|-------------|-------------|-----------|------------------|-----------|------------------|

# Team Elo Ratings
This table summarizes each team's Elo rating and their chances of making it to various stages of the postseason based on 12000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | LAD    |         1562 |            -12 |             -10 | 99.950%    | 99.642%        | 99.950%          | 64.883%    | 37.492%    | 26.242%  |
|  2 | MIL    |         1561 |             -9 |              -1 | 99.858%    | 91.983%        | 99.858%          | 61.867%    | 34.275%    | 23.517%  |
|  3 | NYY    |         1535 |              7 |             -25 | 97.450%    | 40.817%        | 97.450%          | 43.725%    | 27.992%    | 11.933%  |
|  4 | CHC    |         1532 |              1 |              20 | 82.883%    | 7.383%         | 82.883%          | 19.300%    | 8.183%     | 4.550%   |
|  5 | BOS    |         1528 |             11 |              20 | 57.133%    | 2.892%         | 57.133%          | 15.367%    | 8.817%     | 3.475%   |
|  6 | ATL    |         1525 |              3 |             -19 | 91.258%    | 59.925%        | 91.258%          | 21.650%    | 8.425%     | 4.550%   |
|  7 | PHI    |         1524 |             -1 |               2 | 79.275%    | 30.567%        | 79.275%          | 15.867%    | 6.250%     | 3.425%   |
|  8 | TBD    |         1515 |              0 |              10 | 98.375%    | 55.825%        | 98.375%          | 36.667%    | 19.900%    | 6.717%   |
|  9 | CLE    |         1511 |              8 |               0 | 83.000%    | 58.450%        | 83.000%          | 33.500%    | 14.592%    | 4.950%   |
| 10 | SEA    |         1510 |             -6 |              -6 | 54.167%    | 40.608%        | 54.167%          | 17.233%    | 7.467%     | 2.417%   |
| 11 | PIT    |         1510 |              3 |              17 | 23.700%    | 0.133%         | 23.700%          | 2.742%     | 1.025%     | 0.467%   |
| 12 | FLA    |         1509 |             -2 |              14 | 45.275%    | 8.875%         | 45.275%          | 5.958%     | 1.958%     | 0.900%   |
| 13 | DET    |         1505 |              3 |              21 | 14.717%    | 3.850%         | 14.717%          | 3.933%     | 1.617%     | 0.517%   |
| 14 | TOR    |         1503 |              4 |              -7 | 14.917%    | 0.250%         | 14.917%          | 3.067%     | 1.408%     | 0.367%   |
| 15 | ARI    |         1503 |             12 |               7 | 24.292%    | 0.258%         | 24.292%          | 2.533%     | 0.825%     | 0.408%   |
| 16 | STL    |         1501 |              3 |               0 | 32.175%    | 0.492%         | 32.175%          | 3.433%     | 1.075%     | 0.475%   |
| 17 | CHW    |         1500 |              4 |               7 | 65.458%    | 31.408%        | 65.458%          | 19.592%    | 8.167%     | 2.392%   |
| 18 | TEX    |         1498 |             -5 |              -9 | 58.325%    | 45.467%        | 58.325%          | 15.708%    | 6.150%     | 1.608%   |
| 19 | SDP    |         1498 |              1 |             -10 | 13.792%    | 0.100%         | 13.792%          | 1.258%     | 0.400%     | 0.167%   |
| 20 | BAL    |         1494 |              7 |               8 | 11.350%    | 0.217%         | 11.350%          | 2.000%     | 0.750%     | 0.133%   |
| 21 | HOU    |         1488 |             -3 |               2 | 22.692%    | 13.700%        | 22.692%          | 4.858%     | 1.667%     | 0.408%   |
| 22 | MIN    |         1484 |              0 |              24 | 21.983%    | 6.275%         | 21.983%          | 4.333%     | 1.467%     | 0.342%   |
| 23 | WSN    |         1483 |             -2 |              -3 | 7.133%     | 0.633%         | 7.133%           | 0.483%     | 0.092%     | 0.042%   |
| 24 | CIN    |         1477 |              1 |               3 | 0.283%     | 0.008%         | 0.283%           | 0.025%     | <0.025%    | <0.025%  |
| 25 | SFG    |         1476 |             -1 |              -2 | 0.117%     | <0.025%        | 0.117%           | <0.025%    | <0.025%    | <0.025%  |
| 26 | NYM    |         1470 |             -1 |             -23 | 0.008%     | <0.025%        | 0.008%           | <0.025%    | <0.025%    | <0.025%  |
| 27 | KCR    |         1464 |            -12 |             -14 | 0.042%     | 0.017%         | 0.042%           | <0.025%    | <0.025%    | <0.025%  |
| 28 | ANA    |         1455 |              4 |              -6 | 0.042%     | 0.017%         | 0.042%           | <0.025%    | <0.025%    | <0.025%  |
| 29 | OAK    |         1449 |            -13 |             -29 | 0.350%     | 0.208%         | 0.350%           | 0.017%     | 0.008%     | <0.025%  |
| 30 | COL    |         1431 |             -4 |              12 | <0.025%    | <0.025%        | <0.025%          | <0.025%    | <0.025%    | <0.025%  |