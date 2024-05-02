# mlb-elo-advantage
This repository attempts to re-ignite a basic version of FiveThirtyEight's Elo MLB model. The 538 model was shut down in 2023. This repository takes the static Elo ratings from the last day they published and brings them up to speed with today's games in MLB. Special thank you to [Baseball Reference](https://www.baseball-reference.com/) for source of truth game outcomes and schedules. And thank you to [Scores and Odds](https://www.scoresandodds.com/mlb) for scores and odds data (on the nose).

Keeping this Readme pretty high-level to begin with, these were pieces of work:
* Build an Elo model that recreates the basic functionality of 538's per guidance [here](https://www.baseballprospectus.com/news/article/5247/lies-damned-lies-we-are-elo/) [here](https://fivethirtyeight.com/features/how-our-2016-mlb-predictions-work/)
* Populate historical data to collect game outcomes and odds from between when 538 shutdown and when we were ready kick off a daily-updating pipeline
* Build a pipeline that runs every day grabbing the scores and odds from yesterday's games, the odds for today's games, and an updated version of the upcoming schedule
* Based on some backtesting, define a threshold at which picking a team to win could be profitable (hypothetically)
* For each of today's games, use the Elo model to predict the win probabilities and our informed threshold to recommend at what point a team has "value"
