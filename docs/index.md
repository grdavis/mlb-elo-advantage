# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-12 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 4% of games and have a -100.0% ROI over the last 7 days. ROI is -42.15% over the last 30 days and -0.04% over the last 365.

| Date       | Away   | Home   | Away Pitcher   | Home Pitcher    |   Away WinP |   Home WinP |   Away ML | Away Threshold   |   Home ML | Home Threshold   |
|:-----------|:-------|:-------|:---------------|:----------------|------------:|------------:|----------:|:-----------------|----------:|:-----------------|
| 2026-09-12 | COL    | DET    | Tanner Gordon  | Andrew Sears    |       33.23 |       66.77 |       138 | n/a              |      -167 | -162             |
| 2026-09-12 | NYM    | NYY    | Zac Thornton   | Gerrit Cole     |       38.92 |       61.08 |       141 | n/a              |      -170 | -128             |
| 2026-09-12 | PIT    | CHC    | Paul Skenes    | Clay Holmes     |       41.14 |       58.86 |       109 | n/a              |      -132 | -117             |
| 2026-09-12 | BAL    | TOR    | Kyle Bradish   | Spencer Miles   |       44.97 |       55.03 |       109 | +150             |      -131 | -100             |
| 2026-09-12 | SDP    | SFG    | Michael King   | Cesar Perdomo   |       54.94 |       45.06 |      -186 | +100             |       153 | +150             |
| 2026-09-12 | ANA    | WSN    | Walbert Ureña  | Andrew Alvarez  |       45.17 |       54.83 |       105 | +149             |      -127 | +101             |
| 2026-09-12 | KCR    | BOS    | Randy Dobnak   | Ranger Suarez   |       38.7  |       61.3  |       184 | n/a              |      -226 | -129             |
| 2026-09-12 | LAD    | FLA    | Tyler Glasnow  | Tyler Phillips  |       58.5  |       41.5  |      -192 | -115             |       158 | n/a              |
| 2026-09-12 | CLE    | MIN    | Daniel Espino  | Connor Prielipp |       53.35 |       46.65 |      -108 | +107             |      -111 | +140             |
| 2026-09-12 | HOU    | TBD    | Peter Lambert  | Ian Seymour     |       43.42 |       56.58 |       120 | +160             |      -145 | -107             |
| 2026-09-12 | CIN    | MIL    | Brady Singer   |                 |       30.62 |       69.38 |       159 | n/a              |      -193 | -181             |
| 2026-09-12 | PHI    | ATL    |                | Tyler Mahle     |       47    |       53    |       114 | +138             |      -137 | +108             |
| 2026-09-12 | CHW    | STL    | Sean Newcomb   | Kyle Leahy      |       48.12 |       51.88 |      -109 | +132             |      -110 | +113             |
| 2026-09-12 | TEX    | ARI    | Kumar Rocker   | Brandon Pfaadt  |       41.11 |       58.89 |       109 | n/a              |      -131 | -117             |
| 2026-09-12 | SEA    | OAK    | Bryan Woo      | Gage Jump       |       55.46 |       44.54 |      -174 | -102             |       144 | +153             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 23255 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1568 |              5 |              16 | 100.000%   | 100.000%       | 100.000%         | 64.627%    | 37.523%    | 25.491%  |
|  2 | LAD    |         1560 |              9 |              10 | 100.000%   | 100.000%       | 100.000%         | 61.673%    | 31.365%    | 20.305%  |
|  3 | NYY    |         1548 |              4 |              14 | 100.000%   | 24.309%        | 100.000%         | 39.432%    | 28.015%    | 13.227%  |
|  4 | CHC    |         1547 |             -4 |              -5 | 96.753%    | <0.013%        | 96.753%          | 22.141%    | 10.931%    | 6.317%   |
|  5 | BOS    |         1538 |             -6 |             -10 | 99.880%    | 0.022%         | 99.880%          | 22.412%    | 15.167%    | 6.313%   |
|  6 | PHI    |         1535 |             -3 |              17 | 93.404%    | 3.604%         | 93.404%          | 16.280%    | 6.953%     | 3.634%   |
|  7 | ATL    |         1529 |             -4 |             -10 | 100.000%   | 96.396%        | 100.000%         | 20.602%    | 7.942%     | 4.184%   |
|  8 | SDP    |         1527 |              8 |               5 | 64.997%    | <0.013%        | 64.997%          | 9.344%     | 3.556%     | 1.720%   |
|  9 | TBD    |         1527 |              5 |              -5 | 100.000%   | 75.670%        | 100.000%         | 38.400%    | 24.326%    | 9.078%   |
| 10 | ARI    |         1513 |              6 |               2 | 44.588%    | <0.013%        | 44.588%          | 5.311%     | 1.724%     | 0.753%   |
| 11 | TOR    |         1507 |             -8 |               9 | 32.075%    | <0.013%        | 32.075%          | 8.347%     | 3.079%     | 0.976%   |
| 12 | DET    |         1505 |              3 |             -26 | 0.611%     | 0.133%         | 0.611%           | 0.146%     | 0.065%     | 0.026%   |
| 13 | PIT    |         1505 |              5 |               8 | 0.241%     | <0.013%        | 0.241%           | 0.022%     | 0.004%     | 0.004%   |
| 14 | NYM    |         1499 |              7 |              10 | <0.013%    | <0.013%        | <0.013%          | <0.013%    | <0.013%    | <0.013%  |
| 15 | CLE    |         1498 |              3 |               9 | 72.019%    | 40.813%        | 72.019%          | 24.795%    | 8.261%     | 2.137%   |
| 16 | CHW    |         1497 |             -7 |              -6 | 82.064%    | 58.934%        | 82.064%          | 31.610%    | 10.411%    | 2.980%   |
| 17 | HOU    |         1496 |             -2 |              -6 | 91.266%    | 90.062%        | 91.266%          | 30.411%    | 9.435%     | 2.563%   |
| 18 | BAL    |         1496 |              2 |               6 | 6.756%     | <0.013%        | 6.756%           | 1.406%     | 0.482%     | 0.146%   |
| 19 | STL    |         1493 |              0 |              -2 | 0.013%     | <0.013%        | 0.013%           | <0.013%    | <0.013%    | <0.013%  |
| 20 | FLA    |         1490 |             -5 |              -8 | 0.004%     | <0.013%        | 0.004%           | <0.013%    | <0.013%    | <0.013%  |
| 21 | TEX    |         1480 |              0 |              -4 | 12.836%    | 8.484%         | 12.836%          | 2.657%     | 0.675%     | 0.125%   |
| 22 | KCR    |         1480 |              4 |              23 | <0.013%    | <0.013%        | <0.013%          | <0.013%    | <0.013%    | <0.013%  |
| 23 | SEA    |         1475 |             -1 |             -11 | 1.797%     | 1.453%         | 1.797%           | 0.292%     | 0.073%     | 0.017%   |
| 24 | SFG    |         1472 |              1 |              -2 | <0.013%    | <0.013%        | <0.013%          | <0.013%    | <0.013%    | <0.013%  |
| 25 | WSN    |         1471 |             -5 |             -18 | <0.013%    | <0.013%        | <0.013%          | <0.013%    | <0.013%    | <0.013%  |
| 26 | MIN    |         1470 |             -8 |              -8 | 0.697%     | 0.120%         | 0.697%           | 0.090%     | 0.013%     | 0.004%   |
| 27 | CIN    |         1465 |             -7 |             -13 | <0.013%    | <0.013%        | <0.013%          | <0.013%    | <0.013%    | <0.013%  |
| 28 | ANA    |         1454 |              4 |               5 | <0.013%    | <0.013%        | <0.013%          | <0.013%    | <0.013%    | <0.013%  |
| 29 | OAK    |         1435 |              3 |               9 | <0.013%    | <0.013%        | <0.013%          | <0.013%    | <0.013%    | <0.013%  |
| 30 | COL    |         1420 |             -8 |             -10 | <0.013%    | <0.013%        | <0.013%          | <0.013%    | <0.013%    | <0.013%  |