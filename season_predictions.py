import pandas as pd
from random import random
from tqdm import tqdm
import elo

def sim_winner(this_sim, home, away):
	home_winp = this_sim.predict_home_winp(home, away)
	return home if random() < home_winp else away

def finish_season(this_sim, remaining_games):
	'''
	Go through every game in remaining_games, predict the outcome and update the wins and losses in the simulation object
	'''
	for index, row in remaining_games.iterrows():
		winner = sim_winner(this_sim, row['Home'], row['Away'])
		loser = row['Home'] if winner == row['Away'] else row['Away']
		this_sim.teams[winner].season_wins += 1
		this_sim.teams[loser].season_losses += 1

def sim_series(this_sim, home, away, form):
	'''
	Simulates a playoff series between 'home' and 'away' with games in 'form' order and returns a winner.
	form: a string specifying where games are played if the series goes the full length
		e.g. 'hhaah' will be two games at 'home', two at 'away', then 1 back at 'home' (if the series goes all 5 games)
	'''
	wins_needed = len(form) // 2 + 1
	win_dict = {home: 0, away: 0}
	for game in form:
		if game == 'h':
			win_dict[sim_winner(this_sim, home, away)] += 1
		else:
			win_dict[sim_winner(this_sim, away, home)] += 1

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

	For the purposes of this simulation exercise, if there are ties in record, they will be broken randomly

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
	wc_round = [divw['NL'][0], 
					divw['NL'][0], 
					wcs['NL'][0],
					wcs['NL'][1],
					divw['NL'][1],
					divw['NL'][1],
					divw['NL'][2],
					wcs['NL'][2],
					divw['AL'][0], 
					divw['AL'][0], 
					wcs['AL'][0],
					wcs['AL'][1],
					divw['AL'][1],
					divw['AL'][1],
					divw['AL'][2],
					wcs['AL'][2]]

	returns = [divw, wcs]
	#simulate making it to divisional round
	div_round = []
	while len(wc_round) >= 2:
		team1, team2 = wc_round.pop(0), wc_round.pop(0)
		div_round.append(sim_series(this_sim, team1, team2, 'hhh'))
	returns.append(div_round[:])

	#simulate making it to the League Championship Round
	league_round = []
	while len(div_round) >= 2:
		team1, team2 = div_round.pop(0), div_round.pop(0)
		league_round.append(sim_series(this_sim, team1, team2, 'hhaah'))
	returns.append(league_round[:])

	#simulate making it to the WS
	ws_round = []
	while len(league_round) >= 2:
		team1, team2 = league_round.pop(0), league_round.pop(0)
		ws_round.append(sim_series(this_sim, team1, team2, 'hhaaahh'))
	returns.append(ws_round)

	#figure out home team in WS
	if rankings[ws_round[0]] < rankings[ws_round[1]]:
		team1, team2 = ws_round[0], ws_round[1]
	else:
		team2, team1 = ws_round[0], ws_round[1]

	#simulate WS winner
	returns.append(sim_series(this_sim, team1, team2, 'hhaaahh'))

	return returns

def get_playoff_probs(this_sim, game_data):
	'''
	Takes in an Elo simulation and a DataFrame of scores. Orchestrates the running of several simulations on
	the rest of the season and tabulates the probabilities of each team making the playoffs, the CS, DS, WS,
	and winning the whole thing 
	'''
	N_SIMS = 50000
	current_wins = {team: this_sim.teams[team].season_wins for team in this_sim.teams}
	current_losses = {team: this_sim.teams[team].season_losses for team in this_sim.teams}
	remaining_games = game_data[game_data['Home_Score'].isna()]

	#dictionaries mapping team name to counts of occurrences in simulations
	playoffs = {}
	div_wins = {}
	divisional = {}
	championship = {}
	world_series = {}
	ws_winner = {}

	for _ in tqdm(range(N_SIMS)):
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
		for team in outcomes[2]: divisional[team] = divisional.get(team, 0) + 1
		for team in outcomes[3]: championship[team] = championship.get(team, 0) + 1
		for team in outcomes[4]: world_series[team] = world_series.get(team, 0) + 1
		ws_winner[outcomes[5]] = ws_winner.get(outcomes[-1], 0) + 1

	for team in this_sim.teams: playoffs[team] = playoffs.get(team, 0)
	outcomes_df = pd.DataFrame([playoffs, div_wins, divisional, championship, world_series, ws_winner]).T.fillna(0).reset_index()
	outcomes_df.columns = ['Team', 'Playoffs', 'Win Division', 'Reach Div. Rd.', 'Reach CS', 'Reach WS', 'Win WS']
	outcomes_df.iloc[:, 1:] = (outcomes_df.iloc[:, 1:] / N_SIMS).applymap('{:.2%}'.format)
	return outcomes_df