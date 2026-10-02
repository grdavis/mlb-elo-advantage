'''
Regression checks for postseason round simulation.

The 2026-09-30 GitHub page credited a stale hand-entered bracket and treated live
Wild Card games as a later round, so division-round / CS / World Series probabilities
did not add up. These checks lock the derived 12-team bracket, series lengths, and
the per-simulation slot counts (12 playoff teams, 8 LDS, 4 LCS, 2 WS, 1 champion).
'''
import os
import random

import pandas as pd

import season_predictions as sp
import utils

# Regular-season wins through 2026-09-27 (the table the 2026-09-30 page should have used).
WINS_2026 = {
	'CLE': 85, 'MIN': 77, 'DET': 76, 'CHW': 84, 'KCR': 69,
	'TBD': 98, 'NYY': 93, 'BOS': 87, 'BAL': 79, 'TOR': 79,
	'HOU': 81, 'TEX': 80, 'SEA': 76, 'OAK': 64, 'ANA': 62,
	'MIL': 103, 'CHC': 89, 'PIT': 82, 'STL': 77, 'CIN': 75,
	'ATL': 94, 'PHI': 88, 'FLA': 80, 'WSN': 77, 'NYM': 74,
	'LAD': 100, 'SDP': 91, 'ARI': 86, 'SFG': 65, 'COL': 58,
}
# Placeholder bracket that used to be hardcoded. None of DET/CIN/TOR/SEA made the 2026 field.
STALE_PLACEHOLDERS = ('DET', 'CIN', 'TOR', 'SEA')
NL_DIV = ('MIL', 'LAD', 'ATL')
AL_DIV = ('TBD', 'CLE', 'HOU')
NL_WC = ('SDP', 'CHC', 'PHI')
AL_WC = ('NYY', 'BOS', 'CHW')
BYES = ('MIL', 'LAD', 'TBD', 'CLE')


class _Team:
	def __init__(self, name, wins):
		self.season_wins = wins
		self.season_losses = 0
		self.division = utils.TEAM_DIVISIONS[name]
		self.league = self.division[:2]


class _Sim:
	def __init__(self, date, wins, home_winp = 0.5):
		self.date = date
		self.home_winp = home_winp
		self.teams = {name: _Team(name, wins.get(name, 0)) for name in utils.TEAM_DIVISIONS}

	def predict_home_winp(self, home, away, is_playoffs, pitcher_adj = 0):
		return self.home_winp


class _Exploding:
	def predict_home_winp(self, home, away, is_playoffs, pitcher_adj = 0):
		raise AssertionError('series should already be over')


def _game(date, home, away, home_score, away_score):
	return {'Date': date, 'Home': home, 'Away': away, 'Home_Score': home_score, 'Away_Score': away_score}


def make_log(wins, playoff_rows):
	rows = []
	for team, n in wins.items():
		opp = 'OAK' if team == 'COL' else 'COL'
		for _ in range(n):
			rows.append(_game('2026-06-01', team, opp, 1, 0))
	rows.extend(playoff_rows)
	return pd.DataFrame(rows)


def wc_game_one():
	'''Actual 2026-09-29 Wild Card results: higher seeds ATL, NYY, SDP won; CHW beat HOU.'''
	return [
		_game('2026-09-29', 'ATL', 'PHI', 5, 3),
		_game('2026-09-29', 'HOU', 'CHW', 3, 6),
		_game('2026-09-29', 'NYY', 'BOS', 9, 0),
		_game('2026-09-29', 'SDP', 'CHC', 8, 0),
	]


def _fail(msg):
	raise AssertionError(msg)


def _tally(sim, game_data, n, bracket = None):
	counts = {key: {} for key in ('playoffs', 'div', 'lds', 'cs', 'ws', 'champ')}
	for _ in range(n):
		outcomes = sp.setup_playoffs(sim, game_data = game_data, precomputed_bracket = bracket)
		div_teams = [team for league in outcomes[0] for team in outcomes[0][league]]
		wc_teams = [team for league in outcomes[1] for team in outcomes[1][league]]
		lds = [team for team, _w in outcomes[2]]
		cs = [team for team, _w in outcomes[3]]
		ws = [team for team, _w in outcomes[4]]
		champ = outcomes[5]
		if len(div_teams) != 6 or len(set(div_teams)) != 6:
			_fail(f'expected 6 division winners, got {div_teams}')
		if len(wc_teams) != 6 or len(set(wc_teams)) != 6:
			_fail(f'expected 6 wild cards, got {wc_teams}')
		if set(div_teams) & set(wc_teams):
			_fail(f'division winner also listed as a wild card: {set(div_teams) & set(wc_teams)}')
		if len(lds) != 8 or len(set(lds)) != 8:
			_fail(f'expected 8 division-round teams, got {lds}')
		if len(cs) != 4 or len(set(cs)) != 4:
			_fail(f'expected 4 CS teams, got {cs}')
		if len(ws) != 2 or len(set(ws)) != 2:
			_fail(f'expected 2 World Series teams, got {ws}')
		if champ not in ws:
			_fail(f'champion {champ} was not in the World Series {ws}')
		if not set(ws) <= set(cs) <= set(lds) <= (set(div_teams) | set(wc_teams)):
			_fail('later round included a team that had not reached the previous round')
		for team in div_teams:
			counts['div'][team] = counts['div'].get(team, 0) + 1
			counts['playoffs'][team] = counts['playoffs'].get(team, 0) + 1
		for team in wc_teams:
			counts['playoffs'][team] = counts['playoffs'].get(team, 0) + 1
		for team in lds:
			counts['lds'][team] = counts['lds'].get(team, 0) + 1
		for team in cs:
			counts['cs'][team] = counts['cs'].get(team, 0) + 1
		for team in ws:
			counts['ws'][team] = counts['ws'].get(team, 0) + 1
		counts['champ'][champ] = counts['champ'].get(champ, 0) + 1
	return counts


def test_series_lengths():
	boom = _Exploding()
	if sp._wins_needed(sp.WC_FORM) != 2 or sp._wins_needed(sp.LDS_FORM) != 3 or sp._wins_needed(sp.LCS_FORM) != 4:
		_fail('WC/LDS/LCS must be first to 2/3/4')
	if sp.sim_series(boom, 'ATL', 2, 'PHI', 0, sp.WC_FORM) != 'ATL':
		_fail('best-of-3 must end at 2 wins')
	if sp.sim_series(boom, 'MIL', 3, 'SDP', 0, sp.LDS_FORM) != 'MIL':
		_fail('best-of-5 must end at 3 wins')
	if sp.sim_series(boom, 'MIL', 0, 'MIL', 0, sp.WC_FORM) != 'MIL':
		_fail('bye should not play a series')
	try:
		sp.sim_series(boom, 'MIL', 3, 'SDP', 0, sp.LCS_FORM)
	except AssertionError as exc:
		if 'already be over' not in str(exc):
			raise
	else:
		_fail('3 wins must not clinch a best-of-7')
	try:
		sp.sim_series(boom, 'ATL', 1, 'PHI', 0, sp.WC_FORM)
	except AssertionError as exc:
		if 'already be over' not in str(exc):
			raise
	else:
		_fail('a 1-0 Wild Card lead must not be treated as a clinch')
	print('test_series_lengths ok')


def test_locked_field_ignores_stale_bracket_and_playoff_wins():
	log = make_log(WINS_2026, wc_game_one())
	# Poison the sim's win column the way completed playoff games used to.
	sim = _Sim('2026-09-30', WINS_2026)
	sim.teams['CHW'].season_wins = 200
	sim.teams['CLE'].season_wins = 10
	bracket = sp.lock_postseason_bracket(log, sim.date)
	if tuple(bracket['divw']['NL']) != NL_DIV or tuple(bracket['divw']['AL']) != AL_DIV:
		_fail(f"division winners {bracket['divw']}")
	if tuple(bracket['wcs']['NL']) != NL_WC or tuple(bracket['wcs']['AL']) != AL_WC:
		_fail(f"wild cards {bracket['wcs']}")
	for team in STALE_PLACEHOLDERS:
		if team in bracket['seed_of']:
			_fail(f'{team} is in the locked field; that was the stale placeholder')
	# One extra completed CHW win must not reseed AL Central, but it does update the series.
	extra = make_log(WINS_2026, wc_game_one() + [_game('2026-09-30', 'HOU', 'CHW', 1, 2)])
	reseeding = sp.lock_postseason_bracket(extra, '2026-09-30')
	if tuple(reseeding['divw']['AL']) != AL_DIV or tuple(reseeding['wcs']['AL']) != AL_WC:
		_fail(f"playoff win changed the field: {reseeding['divw']} {reseeding['wcs']}")
	hou, chw = sp._indexed_wins(reseeding['win_index'], 'HOU', 'CHW')
	if (hou, chw) != (0, 2):
		_fail(f'expected HOU 0-2 CHW, got {hou}-{chw}')
	outcomes = sp.setup_playoffs(sim, game_data = log)
	lds = {team for team, _w in outcomes[2]}
	if not set(BYES) <= lds:
		_fail(f'bye teams missing from the division round: {set(BYES) - lds}')
	print('test_locked_field_ignores_stale_bracket_and_playoff_wins ok')
	print(sp.format_postseason_status(bracket))


def test_slot_sums_and_wc_best_of_three():
	random.seed(1)
	log = make_log(WINS_2026, wc_game_one())
	sim = _Sim('2026-09-30', {name: 0 for name in WINS_2026}, home_winp = 0.5)
	bracket = sp.lock_postseason_bracket(log, sim.date)
	n = 4000
	counts = _tally(sim, None, n, bracket = bracket)
	if sum(counts['playoffs'].values()) != 12 * n:
		_fail('playoff slot sum')
	if sum(counts['div'].values()) != 6 * n:
		_fail('division slot sum')
	if sum(counts['lds'].values()) != 8 * n:
		_fail('division-round slot sum')
	if sum(counts['cs'].values()) != 4 * n:
		_fail('CS slot sum')
	if sum(counts['ws'].values()) != 2 * n:
		_fail('WS slot sum')
	if sum(counts['champ'].values()) != n:
		_fail('champion slot sum')
	field = set(NL_DIV + AL_DIV + NL_WC + AL_WC)
	if set(counts['playoffs']) != field:
		_fail(f"field {sorted(set(counts['playoffs']))} != {sorted(field)}")
	for team in BYES:
		if counts['lds'].get(team, 0) != n:
			_fail(f'{team} should be in every division round (bye)')
	# 1-0 in a best-of-3 at 50% is 75%. Best-of-5 from 1-0 is 68.75%; ignoring the win is 50%.
	for leader, trailer in (('ATL', 'PHI'), ('SDP', 'CHC'), ('NYY', 'BOS'), ('CHW', 'HOU')):
		rate = counts['lds'].get(leader, 0) / n
		if not 0.72 <= rate <= 0.78:
			_fail(f'{leader} division-round rate {rate:.3f} is not a 1-0 best-of-3 at 50%')
		opp = counts['lds'].get(trailer, 0) / n
		if abs((rate + opp) - 1) > 1e-9:
			_fail(f'{leader}/{trailer} Wild Card rates do not sum to 1 ({rate}, {opp})')
	for team in STALE_PLACEHOLDERS:
		if counts['playoffs'].get(team, 0) != 0:
			_fail(f'stale placeholder {team} made the playoffs')
	print('test_slot_sums_and_wc_best_of_three ok')


def test_clinched_series_advances_only_the_winner():
	sweeps = []
	for home, away in (('ATL', 'PHI'), ('SDP', 'CHC'), ('NYY', 'BOS'), ('HOU', 'CHW')):
		sweeps.append(_game('2026-09-29', home, away, 1, 0))
		sweeps.append(_game('2026-09-30', home, away, 1, 0))
	log = make_log(WINS_2026, sweeps)
	sim = _Sim('2026-09-30', WINS_2026, home_winp = 0.0)
	n = 200
	counts = _tally(sim, log, n)
	reached = {team for team, c in counts['lds'].items() if c == n}
	eliminated = ('PHI', 'CHC', 'BOS', 'CHW')
	expected = set(BYES) | {'ATL', 'SDP', 'NYY', 'HOU'}
	if reached != expected:
		_fail(f'clinched LDS field {sorted(reached)} != {sorted(expected)}')
	for team in eliminated:
		if counts['lds'].get(team, 0) != 0 or counts['cs'].get(team, 0) != 0 or counts['ws'].get(team, 0) != 0:
			_fail(f'{team} advanced after losing a best-of-3')
	print('test_clinched_series_advances_only_the_winner ok')


def test_in_season_simulation_uses_current_standings():
	random.seed(2)
	sim = _Sim('2026-09-01', WINS_2026)
	n = 200
	counts = _tally(sim, None, n)
	if sum(counts['lds'].values()) != 8 * n or sum(counts['ws'].values()) != 2 * n or sum(counts['champ'].values()) != n:
		_fail('in-season round slots do not sum')
	if set(counts['playoffs']) != set(NL_DIV + AL_DIV + NL_WC + AL_WC):
		_fail(f"in-season field drifted: {sorted(counts['playoffs'])}")
	print('test_in_season_simulation_uses_current_standings ok')


def _row_map(table):
	return {row['Team']: row for _, row in table.iterrows()}


def test_exact_zero_when_no_path_remains():
	'''
	Eliminated clubs are exactly 0.00%. The <0.006% floor (3/50,000) stays only
	when the outcome is still reachable and the sample count was zero.
	'''
	if sp.format_outcome_pct(0, 50000, True) != '<0.006%':
		_fail('an open outcome with a zero count must keep the 50,000-sim floor')
	if sp.format_outcome_pct(0, 50000, False) != '0.00%':
		_fail('a closed outcome with a zero count must be 0.00%')
	if sp.format_outcome_pct(25000, 50000, True) != '50.000%':
		_fail('nonzero sample counts stay the observed rate')

	# Games remain, so the field is not locked. Do not invent a magic number.
	open_paths = sp.outcome_paths(None, utils.TEAM_DIVISIONS)
	if open_paths['Win WS'] != set(utils.TEAM_DIVISIONS):
		_fail('in-season title paths should include every team')

	one_game = sp.lock_postseason_bracket(make_log(WINS_2026, wc_game_one()), '2026-09-30')
	one_paths = sp.outcome_paths(one_game, utils.TEAM_DIVISIONS)
	field = set(NL_DIV + AL_DIV + NL_WC + AL_WC)
	if one_paths['Playoffs'] != field:
		_fail(f"playoff paths {sorted(one_paths['Playoffs'])}")
	if one_paths['Win Division'] != set(NL_DIV + AL_DIV):
		_fail('division title paths should be the six winners once the field is locked')
	# A 1-0 lead does not end a best-of-3, so every playoff club can still reach the LDS.
	if one_paths['Reach Div. Rd.'] != field:
		_fail(f"1-0 series dropped {sorted(field - one_paths['Reach Div. Rd.'])}")
	for team in utils.TEAM_DIVISIONS:
		if team in field:
			continue
		for column in sp.OUTCOME_COLUMNS:
			if team in one_paths[column]:
				_fail(f'{team} has a {column} path outside the locked field')

	sweeps = []
	for home, away in (('ATL', 'PHI'), ('SDP', 'CHC'), ('NYY', 'BOS'), ('HOU', 'CHW')):
		sweeps.append(_game('2026-09-29', home, away, 1, 0))
		sweeps.append(_game('2026-09-30', home, away, 1, 0))
	# MIL beats SDP in three games. SDP is out of the LCS; the other LDS series are unplayed.
	sweeps.extend([
		_game('2026-10-03', 'MIL', 'SDP', 1, 0),
		_game('2026-10-04', 'MIL', 'SDP', 1, 0),
		_game('2026-10-06', 'SDP', 'MIL', 0, 1),
	])
	swept = sp.lock_postseason_bracket(make_log(WINS_2026, sweeps), '2026-10-01')
	swept_paths = sp.outcome_paths(swept, utils.TEAM_DIVISIONS)
	for team in ('PHI', 'CHC', 'BOS', 'CHW'):
		if team not in swept_paths['Playoffs'] or team in swept_paths['Reach Div. Rd.']:
			_fail(f'{team} series result was not applied to the division round')
		for column in ('Reach CS', 'Reach WS', 'Win WS'):
			if team in swept_paths[column]:
				_fail(f'{team} still has a path to {column}')
	if 'SDP' not in swept_paths['Reach Div. Rd.'] or 'SDP' in swept_paths['Reach CS']:
		_fail('SDP reached the division round and then lost it')
	if 'MIL' not in swept_paths['Reach CS']:
		_fail('MIL should still be alive for the CS')
	for column in sp.OUTCOME_COLUMNS:
		if not swept_paths[column] <= swept_paths['Playoffs']:
			_fail(f'{column} includes a team outside the playoff field')
	if not swept_paths['Win WS'] <= swept_paths['Reach WS'] <= swept_paths['Reach CS'] <= swept_paths['Reach Div. Rd.']:
		_fail('later rounds are not nested inside earlier rounds')

	# Higher seed wins every remaining game, so a 1-0 trailer is never sampled.
	# The series is not over, so that cell stays the floor. Clubs outside the
	# field, and wild cards for the division title, are exactly 0.00%.
	sim = _Sim('2026-09-30', WINS_2026, home_winp = 1.0)
	n = 30
	table, got_n = sp.get_playoff_probs(sim, make_log(WINS_2026, wc_game_one()), n_sims = n)
	if got_n != n:
		_fail(f'expected {n} sims, got {got_n}')
	floor = '<{:.3f}%'.format(100.0 * 3.0 / n)
	rows = _row_map(table)
	for team in ('COL', 'DET', 'CIN', 'TOR', 'SEA', 'ARI'):
		for column in sp.OUTCOME_COLUMNS:
			if rows[team][column] != '0.00%':
				_fail(f'{team} {column} is {rows[team][column]}, expected 0.00%')
	for team in NL_WC + AL_WC:
		if rows[team]['Playoffs'] != '100.000%' or rows[team]['Win Division'] != '0.00%':
			_fail(f'{team} playoff/division cells {rows[team]["Playoffs"]} {rows[team]["Win Division"]}')
	for team in ('PHI', 'CHC', 'BOS', 'CHW'):
		if rows[team]['Reach Div. Rd.'] != floor:
			_fail(f'{team} division-round cell {rows[team]["Reach Div. Rd."]} should be the floor {floor}')
	if rows['ATL']['Reach Div. Rd.'] != '100.000%' or rows['HOU']['Win Division'] != '100.000%':
		_fail('clinched-or-certain cells should stay at 100%')
	print('test_exact_zero_when_no_path_remains ok')


def test_real_game_log_if_present():
	path = 'DATA/game_log_2026-09-30.csv'
	if not os.path.exists(path):
		print('test_real_game_log_if_present skipped')
		return
	log = pd.read_csv(path)
	bracket = sp.lock_postseason_bracket(log, '2026-09-30')
	if tuple(bracket['divw']['NL']) != NL_DIV or tuple(bracket['wcs']['NL']) != NL_WC:
		_fail(f"real log NL field {bracket['divw']['NL']} {bracket['wcs']['NL']}")
	if tuple(bracket['divw']['AL']) != AL_DIV or tuple(bracket['wcs']['AL']) != AL_WC:
		_fail(f"real log AL field {bracket['divw']['AL']} {bracket['wcs']['AL']}")
	atl, phi = sp._indexed_wins(bracket['win_index'], 'ATL', 'PHI')
	hou, chw = sp._indexed_wins(bracket['win_index'], 'HOU', 'CHW')
	if (atl, phi) != (1, 0) or (hou, chw) != (0, 1):
		_fail(f'real series scores ATL {atl}-{phi} PHI, HOU {hou}-{chw} CHW')
	print('test_real_game_log_if_present ok')


def test_published_october_log_prints_exact_zero():
	'''The 2026-10-01 page showed <0.006% for clubs the locked field had already eliminated.'''
	path = 'DATA/game_log_2026-10-01.csv'
	if not os.path.exists(path):
		print('test_published_october_log_prints_exact_zero skipped')
		return
	log = pd.read_csv(path)
	bracket = sp.lock_postseason_bracket(log, '2026-10-01')
	paths = sp.outcome_paths(bracket, utils.TEAM_DIVISIONS)
	# CHW swept HOU, NYY swept BOS, SDP swept CHC. ATL-PHI is 1-1.
	for team in ('BOS', 'CHC', 'HOU'):
		if team in paths['Reach Div. Rd.']:
			_fail(f'{team} still has a division-round path on the October 1 log')
	for team in ('ATL', 'PHI', 'NYY', 'SDP', 'CHW') + BYES:
		if team not in paths['Reach Div. Rd.']:
			_fail(f'{team} should still be able to reach the division round on October 1')
	if 'PHI' in paths['Win Division'] or 'NYY' in paths['Win Division']:
		_fail('wild cards cannot win a division after the field locks')
	for team in ('COL', 'DET', 'TOR', 'SEA', 'CIN'):
		if team in paths['Playoffs']:
			_fail(f'{team} is outside the field but still has a playoff path')

	sim = _Sim('2026-10-01', {name: 0 for name in utils.TEAM_DIVISIONS}, home_winp = 1.0)
	n = 20
	table, _got = sp.get_playoff_probs(sim, log, n_sims = n)
	rows = _row_map(table)
	floor = '<{:.3f}%'.format(100.0 * 3.0 / n)
	for team in ('COL', 'DET', 'TOR', 'SEA', 'CIN', 'ARI'):
		for column in sp.OUTCOME_COLUMNS:
			if rows[team][column] != '0.00%':
				_fail(f'{team} {column} published as {rows[team][column]}')
	for team in ('BOS', 'CHC'):
		if rows[team]['Playoffs'] != '100.000%' or rows[team]['Reach Div. Rd.'] != '0.00%':
			_fail(f'{team} should be in the playoffs and out of the division round')
		if rows[team]['Win WS'] != '0.00%':
			_fail(f'{team} title cell {rows[team]["Win WS"]}')
	if rows['HOU']['Win Division'] != '100.000%' or rows['HOU']['Reach Div. Rd.'] != '0.00%':
		_fail(f"HOU cells {rows['HOU']['Win Division']} {rows['HOU']['Reach Div. Rd.']}")
	# Game 3 is unplayed, so PHI can still advance. home_winp=1 never samples that.
	if rows['PHI']['Reach Div. Rd.'] != floor or rows['PHI']['Win Division'] != '0.00%':
		_fail(f"PHI cells division {rows['PHI']['Win Division']} LDS {rows['PHI']['Reach Div. Rd.']}")
	if rows['NYY']['Win Division'] != '0.00%' or rows['NYY']['Reach Div. Rd.'] != '100.000%':
		_fail('NYY won the wild card and did not win the division')
	print('test_published_october_log_prints_exact_zero ok')


def main():
	test_series_lengths()
	test_locked_field_ignores_stale_bracket_and_playoff_wins()
	test_slot_sums_and_wc_best_of_three()
	test_clinched_series_advances_only_the_winner()
	test_in_season_simulation_uses_current_standings()
	test_real_game_log_if_present()
	test_exact_zero_when_no_path_remains()
	test_published_october_log_prints_exact_zero()
	print('all playoff regression checks passed')


if __name__ == '__main__':
	main()
