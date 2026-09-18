# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-18 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 5% of games and have a -51.2% ROI over the last 7 days. ROI is -39.48% over the last 30 days and -0.98% over the last 365.

| Date       | Away   | Home   | Away Pitcher    | Home Pitcher      |   Away WinP |   Home WinP |   Away ML | Away Threshold   |   Home ML | Home Threshold   |
|:-----------|:-------|:-------|:----------------|:------------------|------------:|------------:|----------:|:-----------------|----------:|:-----------------|
| 2026-09-18 | CHC    | CIN    | Clay Holmes     | Chase Burns       |       58.88 |       41.12 |      -143 | -117             |       119 | n/a              |
| 2026-09-18 | KCR    | PIT    | Randy Dobnak    | Paul Skenes       |       42.85 |       57.15 |       158 | +164             |      -193 | -109             |
| 2026-09-18 | MIL    | BAL    | Dustin May      | Cade Povich       |       58.24 |       41.76 |      -155 | -114             |       129 | n/a              |
| 2026-09-18 | OAK    | CLE    | Mason Barnett   | Daniel Espino     |       32.21 |       67.79 |       188 | n/a              |      -231 | -169             |
| 2026-09-18 | BOS    | TBD    | Ranger Suarez   | Ian Seymour       |       48.36 |       51.64 |       113 | +131             |      -136 | +114             |
| 2026-09-18 | PHI    | NYM    |                 | Zac Thornton      |       54.8  |       45.2  |      -108 | +101             |      -112 | +149             |
| 2026-09-18 | DET    | CHW    | Andrew Sears    | David Sandlin     |       47.78 |       52.22 |       109 | +134             |      -132 | +112             |
| 2026-09-18 | TOR    | TEX    | Dylan Cease     | Kumar Rocker      |       52.2  |       47.8  |      -137 | +112             |       114 | +134             |
| 2026-09-18 | SEA    | COL    | Bryan Woo       |                   |       57.54 |       42.46 |      -193 | -111             |       158 | n/a              |
| 2026-09-18 | ATL    | HOU    | Tyler Mahle     | Peter Lambert     |       53.94 |       46.06 |      -120 | +104             |       100 | +144             |
| 2026-09-18 | WSN    | STL    | Cade Cavalli    | Kyle Leahy        |       46.34 |       53.66 |      -111 | +142             |      -108 | +106             |
| 2026-09-18 | MIN    | ANA    | Connor Prielipp | Grayson Rodriguez |       48.53 |       51.47 |      -115 | +130             |      -105 | +115             |
| 2026-09-18 | NYY    | ARI    | Gerrit Cole     | Eduardo Rodriguez |       51.09 |       48.91 |      -118 | +117             |      -102 | +128             |
| 2026-09-18 | FLA    | SDP    | Tyler Phillips  | Nick Pivetta      |       40.55 |       59.45 |       179 | n/a              |      -219 | -120             |
| 2026-09-18 | SFG    | LAD    | Cesar Perdomo   | Tyler Glasnow     |       34.57 |       65.43 |       248 | n/a              |      -313 | -153             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 36764 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1568 |              0 |              10 | 100.000%   | 100.000%       | 100.000%         | 63.413%    | 37.463%    | 25.449%  |
|  2 | LAD    |         1559 |             -1 |               8 | 100.000%   | 100.000%       | 100.000%         | 61.441%    | 30.500%    | 19.478%  |
|  3 | CHC    |         1548 |              1 |               3 | 99.736%    | <0.008%        | 99.736%          | 22.780%    | 11.155%    | 6.512%   |
|  4 | NYY    |         1547 |             -1 |              12 | 100.000%   | 5.367%         | 100.000%         | 32.314%    | 23.194%    | 10.804%  |
|  5 | BOS    |         1538 |              0 |             -11 | 99.995%    | <0.008%        | 99.995%          | 20.860%    | 14.171%    | 5.981%   |
|  6 | PHI    |         1536 |              1 |               8 | 99.448%    | 2.089%         | 99.448%          | 17.392%    | 7.475%     | 4.115%   |
|  7 | TBD    |         1536 |              9 |              14 | 100.000%   | 94.633%        | 100.000%         | 46.902%    | 31.066%    | 12.885%  |
|  8 | SDP    |         1531 |              4 |               8 | 95.914%    | <0.008%        | 95.914%          | 15.123%    | 6.074%     | 3.036%   |
|  9 | ATL    |         1528 |             -1 |               1 | 100.000%   | 97.911%        | 100.000%         | 19.429%    | 7.219%     | 3.612%   |
| 10 | DET    |         1511 |              6 |             -11 | 0.092%     | 0.016%         | 0.092%           | 0.030%     | 0.008%     | <0.008%  |
| 11 | TOR    |         1509 |              2 |               6 | 23.050%    | <0.008%        | 23.050%          | 6.104%     | 2.051%     | 0.596%   |
| 12 | ARI    |         1505 |             -8 |              -6 | 4.896%     | <0.008%        | 4.896%           | 0.422%     | 0.114%     | 0.063%   |
| 13 | PIT    |         1505 |              0 |               3 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 14 | CLE    |         1500 |              2 |              11 | 79.303%    | 45.504%        | 79.303%          | 29.719%    | 9.830%     | 2.524%   |
| 15 | BAL    |         1498 |              2 |               4 | 3.705%     | <0.008%        | 3.705%           | 0.862%     | 0.291%     | 0.071%   |
| 16 | CHW    |         1498 |              1 |             -12 | 83.119%    | 54.480%        | 83.119%          | 32.847%    | 10.502%    | 2.747%   |
| 17 | FLA    |         1497 |              7 |              -1 | 0.005%     | <0.008%        | 0.005%           | <0.008%    | <0.008%    | <0.008%  |
| 18 | NYM    |         1494 |             -5 |              -1 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 19 | HOU    |         1493 |             -3 |              -5 | 73.284%    | 69.065%        | 73.284%          | 21.048%    | 6.291%     | 1.510%   |
| 20 | STL    |         1489 |             -4 |             -13 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 21 | TEX    |         1488 |              8 |               9 | 37.352%    | 30.835%        | 37.352%          | 9.297%     | 2.589%     | 0.615%   |
| 22 | SEA    |         1480 |              5 |              -6 | 0.101%     | 0.101%         | 0.101%           | 0.016%     | 0.005%     | 0.003%   |
| 23 | KCR    |         1475 |             -5 |              10 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 24 | SFG    |         1474 |              2 |               6 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 25 | WSN    |         1474 |              3 |             -11 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 26 | CIN    |         1463 |             -2 |             -10 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 27 | MIN    |         1462 |             -8 |             -17 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 28 | ANA    |         1453 |             -1 |               6 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 29 | OAK    |         1426 |             -9 |               1 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |
| 30 | COL    |         1414 |             -6 |             -16 | <0.008%    | <0.008%        | <0.008%          | <0.008%    | <0.008%    | <0.008%  |