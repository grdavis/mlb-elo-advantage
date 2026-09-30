import pandas as pd
from random import random
from tqdm import tqdm
import utils

N_GAMES_TO_SIM = 5000000 # at about 19000 games per second, this targets about a 6 minute runtime no matter how many games remain
# Floor so rare events (e.g. last-place club sneaking into WC) show non-zero rates after rounding
MIN_PLAYOFF_SIMS = 12000
MAX_PLAYOFF_SIMS = 50000

# ============================================================================
# POSTSEASON CALENDAR
# The 12-team field, seeds, and series scores are derived from the game log.
# Only the first postseason date is configured (regular-season rows are strictly
# before it). Unknown years fall back to September 29, the recent norm.
# ============================================================================
PLAYOFF_START_BY_YEAR = {
	2025: '2025-09-29',
	2026: '2026-09-29',
}
PLAYOFF_START_DATE = PLAYOFF_START_BY_YEAR[2026]

# 2022-present round lengths. 'h' is a game at the higher seed.
# Wild Card is best of 3, all at the higher seed.
# LDS is best of 5 (2-2-1). LCS and World Series are best of 7 (2-3-2).
WC_FORM = 'hhh'
LDS_FORM = 'hhaah'
LCS_FORM = 'hhaaahh'
WS_FORM = 'hhaaahh'

def _regular_season_cutoff_for_year(year):
	return PLAYOFF_START_BY_YEAR.get(year, f'{year}-09-29')

def _is_regular_season_date(date_str):
	return date_str < _regular_season_cutoff_for_year(int(date_str[:4]))

def _in_postseason(date_str):
	if not date_str:
		return False
	return date_str >= _regular_season_cutoff_for_year(int(date_str[:4]))

def _wins_needed(form):
	return len(form) // 2 + 1

def sim_winner(this_sim, home, away, is_playoffs, home_winp=None):
	if home_winp is None:
		home_winp = this_sim.predict_home_winp(home, away, is_playoffs)
	return home if random() < home_winp else away

def finish_season(this_sim, remaining_games):
	'''
	Go through every game in remaining_games, predict the outcome and update the wins and losses in the simulation object
	'''
	for index, row in remaining_games.iterrows():
		home_winp = row['HOME_WINP'] if 'HOME_WINP' in remaining_games.columns and pd.notna(row.get('HOME_WINP')) else None
		winner = sim_winner(this_sim, row['Home'], row['Away'], False, home_winp=home_winp)
		loser = row['Home'] if winner == row['Away'] else row['Away']
		this_sim.teams[winner].season_wins += 1
		this_sim.teams[loser].season_losses += 1

def sim_series(this_sim, home, home_wins, away, away_wins, form):
	'''
	Simulates a playoff series between 'home' (higher seed) and 'away'. home_wins and away_wins are the
	current series score so a series in progress is resumed. form is the home/away pattern if the series
	goes the full length, e.g. 'hhaah' is two at home, two at away, then one at home.
	A bye is home == away and returns that team without playing.
	'''
	if home == away:
		return home
	wins_needed = _wins_needed(form)
	if home_wins >= wins_needed:
		return home
	if away_wins >= wins_needed:
		return away
	win_dict = {home: home_wins, away: away_wins}

	for game in form[home_wins + away_wins:]:
		if game == 'h':
			win_dict[sim_winner(this_sim, home, away, is_playoffs = True)] += 1
		else:
			win_dict[sim_winner(this_sim, away, home, is_playoffs = True)] += 1

		if win_dict[home] >= wins_needed:
			return home
		if win_dict[away] >= wins_needed:
			return away
	if win_dict[home] >= win_dict[away]:
		return home
	return away

def _decisive_result(row):
	'''Return (winner, loser, winner_margin) or None if the game has no decisive score.'''
	hs, aws = row['Home_Score'], row['Away_Score']
	if pd.isna(hs) or pd.isna(aws):
		return None
	if str(hs).strip() == '' or str(aws).strip() == '':
		return None
	try:
		hs, aws = float(hs), float(aws)
	except (TypeError, ValueError):
		return None
	if hs == aws:
		return None
	home, away = row['Home'], row['Away']
	if hs > aws:
		return home, away, hs - aws
	return away, home, aws - hs

def rank_group(members, wins, h2h, run_diff):
	'''
	Order teams by wins, then head-to-head wins within the tied group, then run differential, then name.
	Used only to lock the real postseason field. In-season Monte Carlo still breaks ties at random.
	'''
	buckets = {}
	for team in members:
		buckets.setdefault(wins.get(team, 0), []).append(team)
	ranked = []
	for win_total in sorted(buckets, reverse = True):
		group = buckets[win_total]
		def sort_key(team, group = group):
			h2h_wins = sum(h2h.get((team, other), 0) for other in group if other != team)
			return (-h2h_wins, -run_diff.get(team, 0), team)
		ranked.extend(sorted(group, key = sort_key))
	return ranked

def playoff_seeds_from_records(teams, wins, h2h, run_diff):
	'''
	12-team bracket. teams is an iterable of (name, league, division).
	Division winners are re-seeded by record (1-2 bye, 3 plays WC3).
	The three wild cards are seeded 4-6. 4 plays 5, 3 plays 6.
	Returns divw/wcs dicts keyed by league, best record first.
	'''
	by_division = {}
	meta = {}
	for name, league, division in teams:
		by_division.setdefault(division, []).append(name)
		meta[name] = (league, division)
	division_winner = {}
	for division, members in by_division.items():
		division_winner[division] = rank_group(members, wins, h2h, run_diff)[0]

	divw = {'AL': [], 'NL': []}
	wcs = {'AL': [], 'NL': []}
	for league in ('AL', 'NL'):
		winners = [name for name, (lg, division) in meta.items()
			if lg == league and division_winner.get(division) == name]
		# one winner per division; dict order can repeat a winner if meta is odd, so unique it
		seen = []
		for name in winners:
			if name not in seen:
				seen.append(name)
		divw[league] = rank_group(seen, wins, h2h, run_diff)
		rest = [name for name, (lg, division) in meta.items() if lg == league and name not in seen]
		wcs[league] = rank_group(rest, wins, h2h, run_diff)[:3]
	return divw, wcs

def seed_map(divw, wcs):
	seeds = {}
	for league in ('AL', 'NL'):
		for i, team in enumerate(divw[league]):
			seeds[team] = i + 1
		for i, team in enumerate(wcs[league]):
			seeds[team] = i + 4
	return seeds

def series_win_index(playoff_games):
	'''Map frozenset({team_a, team_b}) to {team: series wins} for completed postseason games.'''
	index = {}
	if playoff_games is None or len(playoff_games) == 0:
		return index
	for _, row in playoff_games.iterrows():
		parsed = _decisive_result(row)
		if parsed is None:
			continue
		winner, loser, _margin = parsed
		rec = index.setdefault(frozenset((winner, loser)), {})
		rec[winner] = rec.get(winner, 0) + 1
		rec.setdefault(loser, 0)
	return index

def _indexed_wins(win_index, team_a, team_b):
	rec = win_index.get(frozenset((team_a, team_b)))
	if not rec:
		return 0, 0
	return rec.get(team_a, 0), rec.get(team_b, 0)

def _regular_season_records(game_data, year, playoff_start):
	'''Wins, head-to-head wins, and run differential from this year's regular season only.'''
	wins = {team: 0 for team in utils.TEAM_DIVISIONS}
	run_diff = {team: 0 for team in utils.TEAM_DIVISIONS}
	h2h = {}
	if game_data is None or len(game_data) == 0:
		return wins, h2h, run_diff
	year_prefix = f'{year}-'
	for _, row in game_data.iterrows():
		date = str(row['Date'])
		if not date.startswith(year_prefix) or date >= playoff_start:
			continue
		parsed = _decisive_result(row)
		if parsed is None:
			continue
		winner, loser, margin = parsed
		if winner not in wins or loser not in wins:
			continue
		wins[winner] += 1
		run_diff[winner] += margin
		run_diff[loser] -= margin
		h2h[(winner, loser)] = h2h.get((winner, loser), 0) + 1
	return wins, h2h, run_diff

def _postseason_games(game_data, year, playoff_start):
	if game_data is None or len(game_data) == 0:
		return game_data
	year_prefix = f'{year}-'
	dates = game_data['Date'].astype(str)
	mask = dates.str.startswith(year_prefix) & (dates >= playoff_start)
	return game_data.loc[mask]

def lock_postseason_bracket(game_data, as_of_date):
	'''
	Freeze the playoff field from regular-season results. Playoff wins are not added
	back into the standings, so a Wild Card win cannot change seeds or division titles.
	'''
	year = int(str(as_of_date)[:4])
	playoff_start = _regular_season_cutoff_for_year(year)
	wins, h2h, run_diff = _regular_season_records(game_data, year, playoff_start)
	teams = [(name, division[:2], division) for name, division in utils.TEAM_DIVISIONS.items()]
	divw, wcs = playoff_seeds_from_records(teams, wins, h2h, run_diff)
	ranked = rank_group(list(utils.TEAM_DIVISIONS), wins, h2h, run_diff)
	return {
		'divw': divw,
		'wcs': wcs,
		'seed_of': seed_map(divw, wcs),
		'rank_of': {team: i + 1 for i, team in enumerate(ranked)},
		'win_index': series_win_index(_postseason_games(game_data, year, playoff_start)),
		'wins': wins,
		'playoff_start': playoff_start,
	}

def _seeds_from_sim(this_sim):
	'''Seed a simulated regular-season table. A random fraction breaks exact win ties.'''
	standings = sorted(
		[(team, this_sim.teams[team].season_wins + random(), this_sim.teams[team].league, this_sim.teams[team].division)
			for team in this_sim.teams],
		key = lambda x: x[1], reverse = True)
	div_winners = {}
	wcs = {'AL': [], 'NL': []}
	divw = {'AL': [], 'NL': []}
	rank_of = {}
	rank = 1
	for team, _wins, league, division in standings:
		rank_of[team] = rank
		rank += 1
		if division not in div_winners:
			div_winners[division] = team
			divw[league].append(team)
		elif len(wcs[league]) < 3:
			wcs[league].append(team)
	return divw, wcs, seed_map(divw, wcs), rank_of

def _play(this_sim, high, low, form, win_index):
	if high == low:
		return high
	high_wins, low_wins = _indexed_wins(win_index, high, low)
	return sim_series(this_sim, high, high_wins, low, low_wins, form)

def _home_away_by_seed(team_a, team_b, seed_of):
	'''Better original seed (1 is best) hosts.'''
	if seed_of.get(team_a, 99) <= seed_of.get(team_b, 99):
		return team_a, team_b
	return team_b, team_a

def _home_away_by_rank(team_a, team_b, rank_of):
	'''Better regular-season record hosts the World Series. Lower rank number is better.'''
	if rank_of.get(team_a, 10**9) <= rank_of.get(team_b, 10**9):
		return team_a, team_b
	return team_b, team_a

def simulate_postseason(this_sim, divw, wcs, seed_of, rank_of, win_index):
	'''
	Walk Wild Card -> LDS -> LCS -> World Series. Completed series in win_index are not replayed.
	Each simulated season returns exactly:
	- 6 division winners and 6 wild cards (12 playoff teams)
	- 8 LDS teams (seeds 1-2 plus the four Wild Card winners)
	- 4 LCS teams
	- 2 World Series teams
	- 1 champion
	'''
	lds_entrants = []
	for league in ('NL', 'AL'):
		seed1, seed2, seed3 = divw[league]
		wc1, wc2, wc3 = wcs[league]
		# 1-seed bye, 4 vs 5, 2-seed bye, 3 vs 6. Winners meet in that order in the LDS.
		for high, low in ((seed1, seed1), (wc1, wc2), (seed2, seed2), (seed3, wc3)):
			lds_entrants.append(_play(this_sim, high, low, WC_FORM, win_index))

	lds_winners = []
	for i in range(0, len(lds_entrants), 2):
		lds_winners.append(_play(this_sim, lds_entrants[i], lds_entrants[i + 1], LDS_FORM, win_index))

	nl_home, nl_away = _home_away_by_seed(lds_winners[0], lds_winners[1], seed_of)
	al_home, al_away = _home_away_by_seed(lds_winners[2], lds_winners[3], seed_of)
	nl_champ = _play(this_sim, nl_home, nl_away, LCS_FORM, win_index)
	al_champ = _play(this_sim, al_home, al_away, LCS_FORM, win_index)

	ws_home, ws_away = _home_away_by_rank(nl_champ, al_champ, rank_of)
	ws_winner = _play(this_sim, ws_home, ws_away, WS_FORM, win_index)

	return [
		divw,
		wcs,
		[(team, 0) for team in lds_entrants],
		[(team, 0) for team in (nl_home, nl_away, al_home, al_away)],
		[(team, 0) for team in (ws_home, ws_away)],
		ws_winner,
	]

def setup_playoffs(this_sim, game_data=None, precomputed_bracket=None):
	'''
	Simulate the postseason once. Before the postseason, seeds come from the simulated
	standings. Once the regular season is over, the caller passes a bracket locked from
	regular-season games so playoff results cannot reseed the field.
	'''
	if precomputed_bracket is None and game_data is not None and _in_postseason(this_sim.date):
		precomputed_bracket = lock_postseason_bracket(game_data, this_sim.date)
	if precomputed_bracket is not None:
		return simulate_postseason(
			this_sim,
			precomputed_bracket['divw'],
			precomputed_bracket['wcs'],
			precomputed_bracket['seed_of'],
			precomputed_bracket['rank_of'],
			precomputed_bracket['win_index'],
		)
	divw, wcs, seed_of, rank_of = _seeds_from_sim(this_sim)
	return simulate_postseason(this_sim, divw, wcs, seed_of, rank_of, {})

def _series_line(label, high, low, form, win_index):
	high_wins, low_wins = _indexed_wins(win_index, high, low)
	needed = _wins_needed(form)
	if high_wins >= needed or low_wins >= needed:
		status = 'final'
	elif high_wins + low_wins == 0:
		status = 'not started'
	else:
		status = 'in progress'
	return f'{label} {high} {high_wins}-{low_wins} {low} ({status}, first to {needed})'

def format_postseason_status(bracket):
	'''One-time log of the locked field and Wild Card series state.'''
	lines = [f"[PLAYOFF] Field locked from regular-season games before {bracket['playoff_start']}"]
	wins = bracket['wins']
	for league in ('NL', 'AL'):
		bits = [f"{i + 1}:{team}({wins.get(team, 0)})" for i, team in enumerate(bracket['divw'][league])]
		bits += [f"{i + 4}:{team}({wins.get(team, 0)})" for i, team in enumerate(bracket['wcs'][league])]
		lines.append(f"[PLAYOFF] {league} " + ' '.join(bits))
		seed1, seed2, seed3 = bracket['divw'][league]
		wc1, wc2, wc3 = bracket['wcs'][league]
		lines.append('[PLAYOFF] ' + _series_line(f'{league} WC', wc1, wc2, WC_FORM, bracket['win_index']))
		lines.append('[PLAYOFF] ' + _series_line(f'{league} WC', seed3, wc3, WC_FORM, bracket['win_index']))
		lines.append(f"[PLAYOFF] {league} bye to LDS: {seed1}, {seed2}")
	return '\n'.join(lines)

def get_playoff_probs(this_sim, game_data, n_sims = None):
	'''
	Takes in an Elo simulation and a DataFrame of scores. Orchestrates the running of several simulations on
	the rest of the season and tabulates the probabilities of each team making the playoffs, the CS, DS, WS,
	and winning the whole thing
	'''
	current_wins = {team: this_sim.teams[team].season_wins for team in this_sim.teams}
	current_losses = {team: this_sim.teams[team].season_losses for team in this_sim.teams}
	non_playoffs = game_data[game_data['Date'].map(_is_regular_season_date)]
	remaining_games = non_playoffs[non_playoffs['Home_Score'].isnull() | (non_playoffs['Home_Score'] == '')].copy()
	print(f'Simulating season with {remaining_games.shape[0]} regular season games remaining...')
	# Pregame win probs are static across season sims (Elo is not updated in
	# finish_season). When probable starters are cached, bake the Game Score
	# adjustment in once; otherwise each remaining game uses team Elo only.
	if remaining_games.shape[0] > 0:
		try:
			import pitcher_model
			rg = remaining_games.copy()
			rg['matchup_on_date'] = rg.groupby(['Date', 'Home', 'Away']).cumcount() + 1
			adj_map = pitcher_model.pitcher_adj_lookup(rg)
			winps = []
			for _, row in rg.iterrows():
				adj = adj_map.get((str(row['Date'])[:10], row['Home'], row['Away'], int(row['matchup_on_date'])), 0.0)
				winps.append(this_sim.predict_home_winp(row['Home'], row['Away'], False, pitcher_adj=adj))
			remaining_games['HOME_WINP'] = winps
		except Exception as e:
			print(f'Pitcher-adjusted rest-of-season probs unavailable ({e}); using team Elo only')
			remaining_games['HOME_WINP'] = [
				this_sim.predict_home_winp(r['Home'], r['Away'], False) for _, r in remaining_games.iterrows()
			]

	#dictionaries mapping team name to counts of occurrences in simulations
	playoffs = {}
	div_wins = {}
	divisional = {}
	championship = {}
	world_series = {}
	ws_winner = {}
	if n_sims is None:
		if remaining_games.shape[0] == 0:
			n_sims = MAX_PLAYOFF_SIMS
		else:
			n_sims = min(N_GAMES_TO_SIM // remaining_games.shape[0], MAX_PLAYOFF_SIMS)
			n_sims = max(n_sims, MIN_PLAYOFF_SIMS)

	precomputed_bracket = None
	# Lock only when the regular season is complete. An unplayed pre-cutoff game
	# still has to be simulated, and that can change the field.
	if _in_postseason(this_sim.date) and remaining_games.shape[0] == 0:
		print('\n[PERFORMANCE] Locking playoff bracket from regular-season standings...')
		precomputed_bracket = lock_postseason_bracket(game_data, this_sim.date)
		print(format_postseason_status(precomputed_bracket))

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
	count_df = outcomes_df.iloc[:, 1:7].astype(int)
	# Observed rate with 3 decimals; rule-of-three cap when count==0 (~95% binomial upper bound)
	rule3_pct = 100.0 * 3.0 / n_sims
	def _fmt_pct(c):
		c = int(c)
		if c == 0:
			return '<{:.3f}%'.format(rule3_pct)
		return '{:.3f}%'.format(100.0 * c / n_sims)
	formatted = count_df.apply(lambda col: col.map(_fmt_pct))
	outcomes_df = pd.concat([outcomes_df[['Team']], formatted], axis=1)
	return outcomes_df, n_sims

#Example Testing Query
# import elo
# this_sim, df = elo.main(scrape = False, save_scrape = False, save_new_scrape = False, print_ratings = False)
# outcomes_df, n_sims = get_playoff_probs(this_sim, df)
# print(f"Results based on {n_sims:,} simulated seasons:")
# print(outcomes_df)
