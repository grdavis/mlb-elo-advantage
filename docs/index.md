# MLB Elo Game Predictions and Playoff Probabilities for 2026-09-26 - @grdavis
Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.

Win probabilities include a starting-pitcher Game Score adjustment from the MLB Stats API (team Elo does not know who is on the mound; the market does). Blank pitcher names mean no probable starter is listed yet; that side uses a league-average Game Score of 50. The thresholds are an absolute 5pp edge vs the bettable moneyline, and we do not bet sides longer than +165. Those rules were locked on 2019-2023 data (not the recent losing window). For transparency, these recommendations have been triggered for 8% of games and have a 34.14% ROI over the last 7 days. ROI is -35.96% over the last 30 days and 2.52% over the last 365.

| Date       | Away   | Home   | Away Pitcher    | Home Pitcher     |   Away WinP |   Home WinP | Away ML   | Away Threshold   | Home ML   | Home Threshold   |
|:-----------|:-------|:-------|:----------------|:-----------------|------------:|------------:|:----------|:-----------------|:----------|:-----------------|
| 2026-09-26 | NYM    | WSN    | Jonah Tong      | Connelly Early   |       48.83 |       51.17 | NA        | +128             | NA        | +117             |
| 2026-09-26 | PIT    | DET    | Kirby Yates     | Justin Verlander |       48.55 |       51.45 | NA        | +130             | NA        | +115             |
| 2026-09-26 | CIN    | TOR    | Rhett Lowder    | Trey Yesavage    |       38.63 |       61.37 | 138       | n/a              | -167      | -129             |
| 2026-09-26 | LAD    | SFG    | Blake Snell     | Matt Wilkinson   |       63.37 |       36.63 | -292      | -140             | 233       | n/a              |
| 2026-09-26 | ATL    | FLA    | Brent Suter     | Jack Ralston     |       50.78 |       49.22 | -117      | +118             | -103      | +126             |
| 2026-09-26 | TEX    | MIN    | Nathan Eovaldi  | Bailey Ober      |       48.7  |       51.3  | -125      | +129             | 104       | +116             |
| 2026-09-26 | COL    | CHW    | Jose Quintana   | Davis Martin     |       31.83 |       68.17 | 203       | n/a              | -251      | -172             |
| 2026-09-26 | CLE    | KCR    | Tanner Bibee    | Michael Wacha    |       51.84 |       48.16 | -114      | +114             | -105      | +132             |
| 2026-09-26 | STL    | MIL    | Quinn Mathews   | Dustin May       |       34.72 |       65.28 | 135       | n/a              | -163      | -152             |
| 2026-09-26 | TBD    | PHI    | Griffin Jax     |                  |       49.14 |       50.86 | -108      | +127             | -112      | +118             |
| 2026-09-26 | ARI    | SDP    |                 | Walker Buehler   |       45.53 |       54.47 | 104       | +147             | -126      | +102             |
| 2026-09-26 | HOU    | OAK    | Hayden Wesneski | Jack Perkins     |       56.68 |       43.32 | -161      | -107             | 133       | +161             |
| 2026-09-26 | ANA    | SEA    | Ryan Johnson    | Kade Anderson    |       43.39 |       56.61 | 134       | +161             | -162      | -107             |

# Team Elo Ratings
This table summarizes each team's Elo rating (updated from game results only; pitcher adjustments are pre-game) and their chances of making it to various stages of the postseason based on 50000 simulations of the rest of the regular season and playoffs. Percentages starting with '<' are a rule-of-three upper bound (~95% binomial confidence) when the outcome did not occur in any simulation.

|    | Team   |   Elo Rating |   7-Day Change |   30-Day Change | Playoffs   | Win Division   | Reach Div. Rd.   | Reach CS   | Reach WS   | Win WS   |
|---:|:-------|-------------:|---------------:|----------------:|:-----------|:---------------|:-----------------|:-----------|:-----------|:---------|
|  1 | MIL    |         1577 |              6 |              12 | 100.000%   | 100.000%       | 100.000%         | 67.810%    | 41.732%    | 29.350%  |
|  2 | LAD    |         1563 |              0 |              12 | 100.000%   | 100.000%       | 100.000%         | 65.234%    | 31.746%    | 20.792%  |
|  3 | NYY    |         1546 |             -3 |               9 | 100.000%   | <0.006%        | 100.000%         | 30.714%    | 20.938%    | 9.014%   |
|  4 | CHC    |         1542 |             -6 |              -6 | 96.864%    | <0.006%        | 96.864%          | 16.686%    | 7.460%     | 4.146%   |
|  5 | TBD    |         1537 |              1 |              17 | 100.000%   | 100.000%       | 100.000%         | 48.816%    | 31.462%    | 12.518%  |
|  6 | BOS    |         1537 |             -2 |             -16 | 100.000%   | <0.006%        | 100.000%         | 20.470%    | 13.194%    | 5.100%   |
|  7 | SDP    |         1535 |              0 |               7 | 100.000%   | <0.006%        | 100.000%         | 17.160%    | 7.520%     | 4.022%   |
|  8 | ATL    |         1528 |             -5 |              -4 | 100.000%   | 100.000%       | 100.000%         | 18.598%    | 6.522%     | 3.384%   |
|  9 | PHI    |         1527 |             -2 |              -4 | 85.204%    | <0.006%        | 85.204%          | 12.396%    | 4.404%     | 2.156%   |
| 10 | ARI    |         1517 |             13 |               9 | 17.932%    | <0.006%        | 17.932%          | 2.116%     | 0.616%     | 0.296%   |
| 11 | CLE    |         1509 |              5 |               6 | 100.000%   | 74.220%        | 100.000%         | 48.228%    | 17.546%    | 4.780%   |
| 12 | CHW    |         1508 |             10 |               1 | 100.000%   | 25.780%        | 100.000%         | 32.188%    | 11.386%    | 3.212%   |
| 13 | PIT    |         1508 |              0 |               9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 14 | DET    |         1507 |             -4 |              -5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 15 | TOR    |         1502 |             -1 |               4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 16 | BAL    |         1502 |              6 |               4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 17 | NYM    |         1500 |              0 |               9 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 18 | FLA    |         1498 |              5 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 19 | HOU    |         1489 |              1 |              -3 | 55.526%    | 55.526%        | 55.526%          | 11.278%    | 3.264%     | 0.724%   |
| 20 | TEX    |         1483 |            -11 |               2 | 44.474%    | 44.474%        | 44.474%          | 8.306%     | 2.210%     | 0.506%   |
| 21 | WSN    |         1483 |              1 |               8 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 22 | STL    |         1482 |              1 |             -10 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 23 | MIN    |         1473 |             10 |               3 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 24 | SEA    |         1473 |             -6 |             -13 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 25 | SFG    |         1466 |             -4 |              -4 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 26 | KCR    |         1464 |             -8 |             -17 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 27 | CIN    |         1463 |              0 |              -5 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 28 | ANA    |         1448 |             -4 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 29 | OAK    |         1426 |              4 |              -2 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |
| 30 | COL    |         1407 |             -8 |             -19 | <0.006%    | <0.006%        | <0.006%          | <0.006%    | <0.006%    | <0.006%  |