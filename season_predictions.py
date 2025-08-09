import pandas as pd
from random import random
from tqdm import tqdm
import elo

N_GAMES_TO_SIM = 5000000 # at about 19000 games per second, this targets about a 6 minute runtime no matter how many games remain

def sim_winner(this_sim, home, away, is_playoffs):
	home_winp = this_sim.predict_home_winp(home, away, is_playoffs)
	return home if random() < home_winp else away

def finish_season(this_sim, remaining_games):
	'''
	Go through every game in remaining_games, predict the outcome and update the wins and losses in the simulation object
	'''
	for index, row in remaining_games.iterrows():
		winner = sim_winner(this_sim, row['Home'], row['Away'], False)
		loser = row['Home'] if winner == row['Away'] else row['Away']
		this_sim.teams[winner].season_wins += 1
		this_sim.teams[loser].season_losses += 1

def sim_series(this_sim, home, home_wins, away, away_wins, form):
	'''
	Simulates a playoff series between 'home' and 'away' with games in 'form' order and returns a winner. Also specify
	the current score of the series with home_wins and away_wins, so we can predict the outcome of a series in progress
	form: a string specifying where games are played if the series goes the full length
		e.g. 'hhaah' will be two games at 'home', two at 'away', then 1 back at 'home' (if the series goes all 5 games)
	'''
	wins_needed = len(form) // 2 + 1
	if home_wins == wins_needed: return home
	if away_wins == wins_needed: return away
	win_dict = {home: home_wins, away: away_wins}

	start_index = home_wins + away_wins
	for game in form[start_index:]:
		if game == 'h':
			win_dict[sim_winner(this_sim, home, away, is_playoffs = True)] += 1
		else:
			win_dict[sim_winner(this_sim, away, home, is_playoffs = True)] += 1

		if win_dict[home] == wins_needed: return home
		elif win_dict[away] == wins_needed: return away

def setup_playoffs(this_sim):
	'''
	Take the current win/loss records in this_sim and determine playoff bracket.
	Within each league: #1, #2, #3 are the 3 division winners in record order. #4, #5, #6 are the
	next best records across the league.
	Wild Card Round: #3 plays #6 and #4 plays #5, 3 game series, all 3 at better seed
	Division Series: #1 plays #4/5 and #2 plays #3/6, 5 game series, 2-2-1 format
	Championship Series: 7 game series, 2-3-2 format
	World Series: 7 game series, 2-3-2 format, better record (not necessarily seed) is home

	For the purposes of this simulation exercise, if there are ties in record, they will be broken randomly.
	We don't have all the data in this script to break ties using the actual MLB rules.

	Returns a list consisting of:
		- Dictionary mapping league to a list of teams who won their divisions, in seed order
		- Dictionary mapping league to a list of teams who made the league wild card position, in seed order
		- List of teams in bracket order making it to the divisional round
		- List of teams in bracket order making it to the AL/NLCS
		- List of teams making it to the WS
		- String of WS winner team name
	'''

	#create list of teams and their win counts, add a random number between 0 and 1 to break ties, sort descending
	standings = sorted([(team, this_sim.teams[team].season_wins + random(), this_sim.teams[team].league, this_sim.teams[team].division) for team in this_sim.teams], key = lambda x: x[1], reverse = True)
	
	div_winners = {}
	wcs = {'AL': [], 'NL': []}
	divw = {'AL': [], 'NL': []}
	rankings = {}
	rank = 1
	for team in standings:
		rankings[team[0]] = rank
		if team[3] not in div_winners:
			div_winners[team[3]] = team[0]
			divw[team[2]] += [team[0]]
		elif len(wcs[team[2]]) < 3:
			wcs[team[2]] += [team[0]]
		rank += 1

	#list of 16 teams where pairs of two play each other first round:
	#	NL1, NL1, NL4, NL5, NL2, NL2, NL3, NL6, AL1, AL1, AL4, AL5, AL2, AL2, AL3, AL6
	wc_round = [(divw['NL'][0], 0), 
					(divw['NL'][0], 0),
					(wcs['NL'][0], 0),
					(wcs['NL'][1], 0),
					(divw['NL'][1], 0),
					(divw['NL'][1], 0),
					(divw['NL'][2], 0),
					(wcs['NL'][2], 0),
					(divw['AL'][0], 0),
					(divw['AL'][0], 0),
					(wcs['AL'][0], 0),
					(wcs['AL'][1], 0),
					(divw['AL'][1], 0),
					(divw['AL'][1], 0),
					(divw['AL'][2], 0),
					(wcs['AL'][2], 0)]
	div_round = []
	league_round = []
	ws_round = []

	# update the following with the official playoff bracket as it's released and progresses
	is_current_playoffs = this_sim.date >= '2025-09-29'
	if is_current_playoffs:
		wcs = {'AL': ['BAL', 'KCR', 'DET'], 'NL': ['SDP', 'ATL', 'NYM']}
		divw = {'AL': ['NYY', 'CLE', 'HOU'], 'NL': ['LAD', 'PHI', 'MIL']}
		wc_round = [('LAD', 2), ('LAD', 0), ('SDP', 2), ('ATL', 0), ('PHI', 2), ('PHI', 0), ('MIL', 1), ('NYM', 2), 
					('NYY', 2), ('NYY', 0), ('BAL', 0), ('KCR', 2), ('CLE', 2), ('CLE', 0), ('HOU', 0), ('DET', 2)]
		div_round = [('LAD', 3), ('SDP', 2), ('PHI', 1), ('NYM', 3), 
					('NYY', 3), ('KCR', 1), ('CLE', 3), ('DET', 2)]
		league_round = [('LAD', 4), ('NYM', 2), 
					('NYY', 4), ('CLE', 1)]
		ws_round = [('LAD', 4), ('NYY', 1)]

	returns = [divw, wcs] # start with a list of divisional winners and wild card participants
	#simulate making it to divisional round
	if div_round == []:
		while len(wc_round) >= 2:
			team1, wins1 = wc_round.pop(0)
			team2, wins2 = wc_round.pop(0)
			div_round.append((sim_series(this_sim, team1, wins1, team2, wins2, 'hhh'), 0))
	returns.append(div_round[:])

	#simulate making it to the League Championship Round
	if league_round == []:
		while len(div_round) >= 2:
			team1, wins1 = div_round.pop(0)
			team2, wins2 = div_round.pop(0) 
			league_round.append((sim_series(this_sim, team1, wins1, team2, wins2, 'hhaah'), 0))
	returns.append(league_round[:])

	#simulate making it to the WS
	if ws_round == []:
		while len(league_round) >= 2:
			team1, wins1 = league_round.pop(0)
			team2, wins2 = league_round.pop(0)
			ws_round.append((sim_series(this_sim, team1, wins1, team2, wins2, 'hhaaahh'), 0))
	returns.append(ws_round)

    # figure out home team in WS if not specified already
    # lower ranking number means better record; ensure ws_round has two entries
	if len(ws_round) == 2:
		t1, w1 = ws_round[0]
		t2, w2 = ws_round[1]
		if rankings.get(t1, float('inf')) <= rankings.get(t2, float('inf')):
			team1, wins1 = t1, w1
			team2, wins2 = t2, w2
		else:
			team1, wins1 = t2, w2
			team2, wins2 = t1, w1

	#simulate WS winner
	returns.append(sim_series(this_sim, team1, wins1, team2, wins2, 'hhaaahh'))
	return returns

def get_playoff_probs(this_sim, game_data):
	'''
	Takes in an Elo simulation and a DataFrame of scores. Orchestrates the running of several simulations on
	the rest of the season and tabulates the probabilities of each team making the playoffs, the CS, DS, WS,
	and winning the whole thing 
	'''
	current_wins = {team: this_sim.teams[team].season_wins for team in this_sim.teams}
	current_losses = {team: this_sim.teams[team].season_losses for team in this_sim.teams}
	non_playoffs = game_data[game_data['Date'] < '2025-09-29']
	remaining_games = non_playoffs[non_playoffs['Home_Score'].isnull() | (non_playoffs['Home_Score'] == '')]
	print(f'Simulating season with {remaining_games.shape[0]} regular season games remaining...')

	#dictionaries mapping team name to counts of occurrences in simulations
	playoffs = {}
	div_wins = {}
	divisional = {}
	championship = {}
	world_series = {}
	ws_winner = {}
	n_sims = min(N_GAMES_TO_SIM // remaining_games.shape[0], 50000) # calculate how many seasons we can simulate

	for _ in tqdm(range(n_sims)):
		#reset the win and loss counts to what they are currently at the start of every simulation
		for team in current_wins:
			this_sim.teams[team].season_wins = current_wins[team]
			this_sim.teams[team].season_losses = current_losses[team]

		finish_season(this_sim, remaining_games)
		outcomes = setup_playoffs(this_sim)
		
		#add appearances in each of these rounds to the overall trackers
		for league in outcomes[0]:
			for team in outcomes[0][league]:
				playoffs[team] = playoffs.get(team, 0) + 1
				div_wins[team] = div_wins.get(team, 0) + 1
		for league in outcomes[1]:
			for team in outcomes[1][league]:
				playoffs[team] = playoffs.get(team, 0) + 1
		for team, wins in outcomes[2]: divisional[team] = divisional.get(team, 0) + 1
		for team, wins in outcomes[3]: championship[team] = championship.get(team, 0) + 1
		for team, wins in outcomes[4]: world_series[team] = world_series.get(team, 0) + 1
		ws_winner[outcomes[5]] = ws_winner.get(outcomes[-1], 0) + 1

	for team in this_sim.teams: playoffs[team] = playoffs.get(team, 0)
	outcomes_df = pd.DataFrame([playoffs, div_wins, divisional, championship, world_series, ws_winner]).T.fillna(0).reset_index()
	outcomes_df.columns = ['Team', 'Playoffs', 'Win Division', 'Reach Div. Rd.', 'Reach CS', 'Reach WS', 'Win WS']
	outcomes_df.iloc[:, 1:] = (outcomes_df.iloc[:, 1:] / n_sims).applymap('{:.2%}'.format)
	return outcomes_df, n_sims

#Example Testing Query
# this_sim, df = elo.main(scrape = False, save_scrape = False, save_new_scrape = False, print_ratings = False)
# outcomes_df, n_sims = get_playoff_probs(this_sim, df)
# print(f"Results based on {n_sims:,} simulated seasons:")
# print(outcomes_df)