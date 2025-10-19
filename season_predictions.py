import pandas as pd
from random import random
from tqdm import tqdm

N_GAMES_TO_SIM = 5000000 # at about 19000 games per second, this targets about a 6 minute runtime no matter how many games remain

# ============================================================================
# 2025 PLAYOFF BRACKET CONFIGURATION (ONE-TIME SETUP)
# Update this section when playoffs start each year, then never touch it again!
# ============================================================================
PLAYOFF_START_DATE = '2025-09-29'
PLAYOFF_WILD_CARDS = {'AL': ['NYY', 'BOS', 'DET'], 'NL': ['CHC', 'SDP', 'CIN']}
PLAYOFF_DIV_WINNERS = {'AL': ['TOR', 'SEA', 'CLE'], 'NL': ['MIL', 'PHI', 'LAD']}

# Initial Wild Card bracket structure (determines who plays whom)
# Pattern: Division winners play against wild cards in this order
# Each pair of entries represents one matchup (better seed listed first)
PLAYOFF_WC_BRACKET = [
	# NL matchups (indices 0-7)
	('MIL', 0), ('MIL', 0), ('CHC', 0), ('SDP', 0),  # MIL(1) gets bye, WC3 vs WC4
	('PHI', 0), ('PHI', 0), ('LAD', 0), ('CIN', 0),  # PHI(2) gets bye, WC5 vs WC6
	# AL matchups (indices 8-15)
	('TOR', 0), ('TOR', 0), ('NYY', 0), ('BOS', 0),  # TOR(1) gets bye, WC3 vs WC4
	('SEA', 0), ('SEA', 0), ('CLE', 0), ('DET', 0)   # SEA(2) gets bye, WC5 vs WC6
]
# ============================================================================

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

def count_series_wins(playoff_games, team1, team2):
	'''Count wins for team1 and team2 in their head-to-head series.'''
	matchup_games = playoff_games[
		((playoff_games['Home'] == team1) & (playoff_games['Away'] == team2)) |
		((playoff_games['Home'] == team2) & (playoff_games['Away'] == team1))
	]
	
	team1_wins, team2_wins = 0, 0
	for _, game in matchup_games.iterrows():
		winner = game['Home'] if float(game['Home_Score']) > float(game['Away_Score']) else game['Away']
		if winner == team1:
			team1_wins += 1
		else:
			team2_wins += 1
	return team1_wins, team2_wins

def update_playoff_series_from_games(game_data, wc_round, playoff_start_date='2025-09-29'):
	'''Parse game log, update all series scores, auto-detect advancing matchups.'''
	playoff_games = game_data[
		(game_data['Date'] >= playoff_start_date) & 
		(game_data['Home_Score'].notna()) &
		(game_data['Home_Score'] != '')
	].copy()
	
	if len(playoff_games) == 0:
		return wc_round, [], [], []
	
	print(f'[PLAYOFF PARSER] Found {len(playoff_games)} completed playoff games')
	
	wc_pairs = {tuple(sorted([wc_round[i][0], wc_round[i+1][0]])) 
	            for i in range(0, len(wc_round), 2)}
	
	# Build dict of matchups with their first game date for chronological sorting
	all_matchups = {}
	for _, game in playoff_games.iterrows():
		pair = tuple(sorted([game['Home'], game['Away']]))
		if pair not in all_matchups:
			all_matchups[pair] = {'teams': (game['Home'], game['Away']), 'first_game': game['Date']}
	
	def build_round(matchup_pairs, wins_needed):
		round_list = []
		for team1, team2 in matchup_pairs:
			wins1, wins2 = count_series_wins(playoff_games, team1, team2)
			round_list.extend([(team1, wins1), (team2, wins2)])
			if wins1 > 0 or wins2 > 0:
				status = 'complete' if max(wins1, wins2) >= wins_needed else 'in progress'
				print(f'[PLAYOFF PARSER] {team1} {wins1}-{wins2} {team2} ({status})')
		return round_list
	
	# Separate matchups by round using chronological order
	# Sort all non-WC matchups by first game date
	non_wc_matchups = [(p, info) for p, info in all_matchups.items() if p not in wc_pairs]
	non_wc_matchups.sort(key=lambda x: x[1]['first_game'])
	
	# Wild Card round
	updated_wc_round = build_round([all_matchups[p]['teams'] for p in wc_pairs if p in all_matchups], 2)
	
	# Division Series: next 4 matchups chronologically (2 per league)
	div_matchups = non_wc_matchups[:4] if len(non_wc_matchups) >= 4 else non_wc_matchups
	updated_div_round = build_round([info['teams'] for p, info in div_matchups], 3)
	
	# Championship Series: next 2 matchups chronologically (1 per league)
	cs_matchups = non_wc_matchups[4:6] if len(non_wc_matchups) >= 6 else []
	updated_league_round = build_round([info['teams'] for p, info in cs_matchups], 4)
	
	# World Series: next 1 matchup chronologically
	ws_matchups = non_wc_matchups[6:7] if len(non_wc_matchups) >= 7 else []
	updated_ws_round = build_round([info['teams'] for p, info in ws_matchups], 4)
	
	return updated_wc_round, updated_div_round, updated_league_round, updated_ws_round

def setup_playoffs(this_sim, game_data=None, precomputed_bracket=None):
	'''Determine playoff bracket from standings or use precomputed bracket. Simulates series to find WS winner.'''

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

	# Setup for current playoffs
	is_current_playoffs = this_sim.date >= PLAYOFF_START_DATE
	if is_current_playoffs:
		# Get initial bracket state (either from precomputed or from game log)
		if precomputed_bracket is not None:
			wcs = precomputed_bracket['wcs']
			divw = precomputed_bracket['divw']
			wc_round = precomputed_bracket['wc_round'][:]  # Copy to avoid mutation
			div_round = precomputed_bracket['div_round'][:]
			league_round = precomputed_bracket['league_round'][:]
			ws_round = precomputed_bracket['ws_round'][:]
		else:
			wcs = PLAYOFF_WILD_CARDS.copy()
			divw = PLAYOFF_DIV_WINNERS.copy()
			wc_round = PLAYOFF_WC_BRACKET.copy()
			if game_data is not None:
				wc_round, div_round, league_round, ws_round = update_playoff_series_from_games(
					game_data, wc_round, PLAYOFF_START_DATE
				)
			else:
				div_round, league_round, ws_round = [], [], []

	returns = [divw, wcs]
	
	# Simulate Division Round from current state (ALWAYS simulate, even with precomputed bracket)
	div_round_participants = []
	if div_round == []:
		# Build from WC winners
		while len(wc_round) >= 2:
			team1, wins1 = wc_round.pop(0)
			team2, wins2 = wc_round.pop(0)
			winner = sim_series(this_sim, team1, wins1, team2, wins2, 'hhh')
			div_round.append((winner, 0))
			# Only add unique teams (handles byes where team1 == team2)
			if team1 == team2:
				div_round_participants.append((team1, wins1))
			else:
				div_round_participants.extend([(team1, wins1), (team2, wins2)])
	else:
		# Division round in progress - simulate each series from current state
		div_round_participants = div_round[:]  # Track all participants for stats
		winners = []
		for i in range(0, len(div_round), 2):
			if i+1 < len(div_round):
				team1, wins1 = div_round[i]
				team2, wins2 = div_round[i+1]
				winner = sim_series(this_sim, team1, wins1, team2, wins2, 'hhaah')
				winners.append((winner, 0))
		div_round = winners  # Use winners for next round simulation
	returns.append(div_round_participants)

	# Simulate League Championship Round
	league_round_participants = []
	if league_round == []:
		while len(div_round) >= 2:
			team1, wins1 = div_round.pop(0)
			team2, wins2 = div_round.pop(0)
			winner = sim_series(this_sim, team1, wins1, team2, wins2, 'hhaaahh')
			league_round.append((winner, 0))
			league_round_participants.extend([(team1, wins1), (team2, wins2)])
	elif len(league_round) >= 2:
		# League round already started - track participants and simulate to completion
		league_round_participants = league_round[:]
		winners = []
		for i in range(0, len(league_round), 2):
			if i+1 < len(league_round):
				team1, wins1 = league_round[i]
				team2, wins2 = league_round[i+1]
				winner = sim_series(this_sim, team1, wins1, team2, wins2, 'hhaaahh')
				winners.append((winner, 0))
		league_round = winners
	returns.append(league_round_participants)

	# Build World Series matchup (don't simulate yet - that happens at the end)
	ws_round_participants = []
	if ws_round == []:
		# Advance winners from league championship
		while len(league_round) >= 2:
			team1, wins1 = league_round.pop(0)
			team2, wins2 = league_round.pop(0)
			ws_round.extend([(team1, wins1), (team2, wins2)])
			ws_round_participants.extend([(team1, wins1), (team2, wins2)])
	elif len(ws_round) >= 2:
		# WS already started
		ws_round_participants = ws_round[:]
	returns.append(ws_round_participants)

	# Simulate WS winner if we have both participants
	if len(ws_round) >= 2:
		# WS participants are in ws_round, simulate to get winner
		team1, wins1 = ws_round[0]
		team2, wins2 = ws_round[1]
		# Better record gets home field
		if rankings.get(team1, float('inf')) <= rankings.get(team2, float('inf')):
			ws_winner = sim_series(this_sim, team1, wins1, team2, wins2, 'hhaaahh')
		else:
			ws_winner = sim_series(this_sim, team2, wins2, team1, wins1, 'hhaaahh')
		returns.append(ws_winner)
	else:
		returns.append(None)  # No WS winner yet
	
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
	if remaining_games.shape[0] == 0:
		n_sims = 50000  # Use full simulation budget when no games remain
	else:
		n_sims = min(N_GAMES_TO_SIM // remaining_games.shape[0], 50000) # calculate how many seasons we can simulate

	precomputed_bracket = None
	if this_sim.date >= PLAYOFF_START_DATE:
		wcs = PLAYOFF_WILD_CARDS.copy()
		divw = PLAYOFF_DIV_WINNERS.copy()
		print('\n[PERFORMANCE] Pre-computing playoff bracket from game log...')
		wc_round, div_round, league_round, ws_round = update_playoff_series_from_games(
			game_data, PLAYOFF_WC_BRACKET.copy(), PLAYOFF_START_DATE
		)
		
		precomputed_bracket = {
			'wcs': wcs,
			'divw': divw,
			'wc_round': wc_round,
			'div_round': div_round,
			'league_round': league_round,
			'ws_round': ws_round
		}

	for _ in tqdm(range(n_sims)):
		#reset the win and loss counts to what they are currently at the start of every simulation
		for team in current_wins:
			this_sim.teams[team].season_wins = current_wins[team]
			this_sim.teams[team].season_losses = current_losses[team]

		finish_season(this_sim, remaining_games)
		outcomes = setup_playoffs(this_sim, game_data=None, precomputed_bracket=precomputed_bracket)
		
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
		if outcomes[5] is not None:
			ws_winner[outcomes[5]] = ws_winner.get(outcomes[5], 0) + 1

	for team in this_sim.teams: playoffs[team] = playoffs.get(team, 0)
	outcomes_df = pd.DataFrame([playoffs, div_wins, divisional, championship, world_series, ws_winner]).T.fillna(0).reset_index()
	outcomes_df.columns = ['Team', 'Playoffs', 'Win Division', 'Reach Div. Rd.', 'Reach CS', 'Reach WS', 'Win WS']
	outcomes_df.iloc[:, 1:] = (outcomes_df.iloc[:, 1:] / n_sims).applymap('{:.2%}'.format)
	return outcomes_df, n_sims

#Example Testing Query
# import elo
# this_sim, df = elo.main(scrape = False, save_scrape = False, save_new_scrape = False, print_ratings = False)
# outcomes_df, n_sims = get_playoff_probs(this_sim, df)
# print(f"Results based on {n_sims:,} simulated seasons:")
# print(outcomes_df)