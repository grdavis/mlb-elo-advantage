# MLB Elo Game Predictions and Playoff Probabilities for 2026-07-16 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

The thresholds indicate at what odds the model thinks there is value in betting on a team. These thresholds were selected via backtesting over the last 2 years (2.7K games). For transparency, these recommendations have been triggered for 22% of games and have a 18.19% ROI over the last 7 days. ROI is -13.9% over the last 30 days and -2.53% over the last 365.

| Date       | Away   | Home   |   Away WinP |   Home WinP |   Away ML |   Away Threshold |   Home ML |   Home Threshold |
|:-----------|:-------|:-------|------------:|------------:|----------:|-----------------:|----------:|-----------------:|
| 2026-07-16 | NYM    | PHI    |       39.51 |       60.49 |       109 |             +181 |      -131 |             -120 |

# Team Elo Ratings
This table summarizes each team's Elo rating and their chances of making it to various stages of the postseason based on 12000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | LAD    |         1562 |            -12 |             -12 | 99.933%    | 99.683%        | 99.933%          | 64.683%    | 37.000%    | 25.717%  |
|  2 | MIL    |         1561 |             -8 |              -2 | 99.858%    | 92.117%        | 99.858%          | 61.817%    | 34.058%    | 23.350%  |
|  3 | NYY    |         1535 |              5 |             -28 | 97.350%    | 40.642%        | 97.350%          | 43.525%    | 26.933%    | 11.492%  |
|  4 | CHC    |         1532 |              1 |              22 | 82.983%    | 7.167%         | 82.983%          | 19.508%    | 8.225%     | 4.550%   |
|  5 | BOS    |         1528 |              7 |              23 | 57.750%    | 3.167%         | 57.750%          | 16.200%    | 9.125%     | 3.775%   |
|  6 | ATL    |         1525 |             -2 |             -15 | 91.467%    | 59.225%        | 91.467%          | 20.725%    | 8.383%     | 4.517%   |
|  7 | PHI    |         1524 |              1 |              -4 | 79.167%    | 31.242%        | 79.167%          | 15.733%    | 6.708%     | 3.517%   |
|  8 | TBD    |         1515 |              2 |              12 | 98.417%    | 55.825%        | 98.417%          | 36.750%    | 19.908%    | 7.008%   |
|  9 | CLE    |         1511 |              7 |               1 | 83.075%    | 58.258%        | 83.075%          | 33.258%    | 14.767%    | 5.167%   |
| 10 | SEA    |         1510 |             -1 |              -8 | 53.700%    | 39.925%        | 53.700%          | 17.117%    | 7.642%     | 2.417%   |
| 11 | PIT    |         1510 |              8 |              20 | 23.575%    | 0.300%         | 23.575%          | 3.192%     | 1.025%     | 0.458%   |
| 12 | FLA    |         1509 |             -7 |              20 | 45.600%    | 8.900%         | 45.600%          | 6.167%     | 2.050%     | 0.875%   |
| 13 | DET    |         1505 |             -1 |              19 | 15.217%    | 4.100%         | 15.217%          | 4.117%     | 2.000%     | 0.550%   |
| 14 | TOR    |         1503 |             -1 |             -10 | 14.867%    | 0.242%         | 14.867%          | 2.742%     | 1.192%     | 0.267%   |
| 15 | ARI    |         1503 |             13 |              10 | 25.000%    | 0.275%         | 25.000%          | 2.858%     | 0.942%     | 0.458%   |
| 16 | STL    |         1501 |              2 |              -4 | 31.333%    | 0.417%         | 31.333%          | 3.442%     | 1.092%     | 0.442%   |
| 17 | CHW    |         1500 |              9 |              10 | 64.433%    | 30.750%        | 64.433%          | 18.767%    | 7.992%     | 2.300%   |
| 18 | TEX    |         1498 |              0 |              -1 | 59.383%    | 46.617%        | 59.383%          | 16.125%    | 6.408%     | 2.000%   |
| 19 | SDP    |         1498 |              0 |              -6 | 13.908%    | 0.042%         | 13.908%          | 1.417%     | 0.417%     | 0.183%   |
| 20 | BAL    |         1494 |              7 |               9 | 11.433%    | 0.125%         | 11.433%          | 1.925%     | 0.867%     | 0.225%   |
| 21 | HOU    |         1488 |              0 |               4 | 22.300%    | 13.158%        | 22.300%          | 4.733%     | 1.525%     | 0.333%   |
| 22 | MIN    |         1484 |              1 |              16 | 21.667%    | 6.892%         | 21.667%          | 4.692%     | 1.625%     | 0.375%   |
| 23 | WSN    |         1483 |             -5 |              -7 | 6.925%     | 0.633%         | 6.925%           | 0.450%     | 0.100%     | 0.025%   |
| 24 | CIN    |         1477 |             -2 |              -4 | 0.208%     | <0.025%        | 0.208%           | <0.025%    | <0.025%    | <0.025%  |
| 25 | SFG    |         1476 |              1 |              -6 | 0.033%     | <0.025%        | 0.033%           | 0.008%     | <0.025%    | <0.025%  |
| 26 | NYM    |         1470 |             -6 |             -16 | 0.008%     | <0.025%        | 0.008%           | <0.025%    | <0.025%    | <0.025%  |
| 27 | KCR    |         1464 |             -7 |             -10 | 0.008%     | <0.025%        | 0.008%           | <0.025%    | <0.025%    | <0.025%  |
| 28 | ANA    |         1455 |             -1 |              -9 | 0.025%     | 0.017%         | 0.025%           | <0.025%    | <0.025%    | <0.025%  |
| 29 | OAK    |         1449 |             -9 |             -32 | 0.375%     | 0.283%         | 0.375%           | 0.050%     | 0.017%     | <0.025%  |
| 30 | COL    |         1431 |             -1 |              10 | <0.025%    | <0.025%        | <0.025%          | <0.025%    | <0.025%    | <0.025%  |