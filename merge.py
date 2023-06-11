import pandas as pd
from datetime import datetime

elo_file = 'mlb_elo.csv'
logs_file = 'game_logs_2019_to_20230605.csv'
save_destination = 'cleaned_logs_2019_to_20230605.csv'
start_date = '2019-03-20'
end_date = '2023-06-05'

TEAM_MAP = {
	'Chi Cubs': 'CHC',
	'Atlanta': 'ATL',
	'Baltimore': 'BAL',
	'Cleveland': 'CLE',
	'NY Yankees': 'NYY',
	'Tampa Bay': 'TBD',
	'Toronto': 'TOR',
	'Colorado': 'COL',
	'LA Angels': 'ANA',
	'Oakland': 'OAK',
	'Milwaukee': 'MIL',
	'Philadelphia': 'PHI',
	'Seattle': 'SEA',
	'St. Louis': 'STL',
	'Detroit': 'DET',
	'Cincinnati': 'CIN',
	'San Diego': 'SDP',
	'Minnesota': 'MIN',
	'Washington': 'WSN',
	'Pittsburgh': 'PIT',
	'Texas': 'TEX',
	'Kansas City': 'KCR',
	'Boston': 'BOS',
	'Houston': 'HOU',
	'Miami': 'FLA',
	'NY Mets': 'NYM',
	'San Francisco': 'SFG',
	'Chi White Sox': 'CHW',
	'LA Dodgers': 'LAD',
	'Arizona': 'ARI'
}

def fix_datestring(ds):
	split_ds = ds.split(' ')
	new_ds = f'{split_ds[0]} 0{split_ds[1]} {split_ds[2]}' if len(split_ds[1]) == 1 else ds
	return datetime.strptime(new_ds, '%b %d, %Y')

###import and clean up the game logs data###
ldf = pd.read_csv(logs_file, header = None, names = ['T1', 'DATE', 'T2', 'STYPE', 'T1_RESULT', 'SCORE', 'T1_ML', 'OU_RESULT', 'OU_LINE'])

#define the home and away teams, split out the scores to home and away scores, clean up game date format
aflags, hscores, ascores, gdates, hteam, ateam = [], [], [], [], [], []
for index, row in ldf.iterrows():
	if row['T2'][0] == '@':
		hteam.append(TEAM_MAP[row['T2'][2:]])
		ateam.append(row['T1'])
	else:
		hteam.append(row['T1'])
		ateam.append(TEAM_MAP[row['T2'][3:]])
	aflags.append(row['T2'][0] == '@')
	gdates.append(fix_datestring(row['DATE']))
	if pd.isna(row['SCORE']):
		ascores.append('')
		hscores.append('')
	else:
		winner, loser = row['SCORE'].split('-')
		if row['T2'][0] == '@': 
			ascores.append(winner if row['T1_RESULT'] == 'W' else loser)
			hscores.append(winner if row['T1_RESULT'] == 'L' else loser)
		else:
			hscores.append(winner if row['T1_RESULT'] == 'W' else loser)
			ascores.append(winner if row['T1_RESULT'] == 'L' else loser)

ldf['arow'] = aflags
ldf['HOME_SCORE'], ldf['AWAY_SCORE'] = hscores, ascores
ldf['GAMEDATE'] = gdates
ldf['HOME'], ldf['AWAY'] = hteam, ateam

#trim to just games that have occurred
ldf = ldf.loc[ldf['GAMEDATE'] <= end_date]

#dataframe contains each game twice, merge the two instances now that we know home and away (we needed to preserve them both to get both team's ML odds)
ldf_home = ldf.loc[ldf['arow'] == False].reset_index(drop = True)
ldf_away = ldf.loc[ldf['arow'] == True].reset_index(drop = True)
ldf = ldf_home.merge(ldf_away, how = 'inner', on = ['GAMEDATE', 'HOME', 'AWAY', 'HOME_SCORE', 'AWAY_SCORE', 'OU_LINE', 'OU_RESULT'])
ldf = ldf[['GAMEDATE', 'HOME', 'AWAY', 'HOME_SCORE', 'AWAY_SCORE', 'T1_ML_x', 'T1_ML_y', 'OU_RESULT', 'OU_LINE', 'arow_x']]
ldf['HOME_ML'] = ldf.apply(lambda x: x['T1_ML_x'] if not x['arow_x'] else x['T1_ML_y'], axis = 1)
ldf['AWAY_ML'] = ldf.apply(lambda x: x['T1_ML_x'] if x['arow_x'] else x['T1_ML_y'], axis = 1)


###import and clean up the ELO data###
edf = pd.read_csv(elo_file)

#reformat game date and trim to just relevant games that could be matched
edf['GAMEDATE'] = edf.apply(lambda x: datetime.strptime(x['date'], '%Y-%m-%d'), axis = 1)
edf['score1'] = edf.apply(lambda x: str(int(x['score1'])) if not pd.isna(x['score1']) else '', axis = 1)
edf['score2'] = edf.apply(lambda x: str(int(x['score2'])) if not pd.isna(x['score2']) else '', axis = 1)
edf = edf.loc[(edf['GAMEDATE'] >= start_date) & (edf['GAMEDATE'] <= end_date)]


###merge the ELO and game log data###
combo = edf.merge(ldf, how = 'inner', left_on = ['team1', 'team2', 'GAMEDATE', 'score1', 'score2'], right_on = ['HOME', 'AWAY', 'GAMEDATE', 'HOME_SCORE', 'AWAY_SCORE'])
combo = combo[['GAMEDATE', 'HOME', 'AWAY', 'HOME_SCORE', 'AWAY_SCORE', 'HOME_ML', 'AWAY_ML', 
				'rating1_pre', 'rating2_pre', 'rating_prob1', 'rating_prob2', 'OU_LINE', 'OU_RESULT']]
combo.rename(columns={'rating_prob1':'HOME_PROB', 'rating_prob2':'AWAY_PROB', 'rating1_pre':'HOME_RATING', 'rating2_pre':'AWAY_RATING'}, inplace=True)


###perform some cleaning and enriching on the merged dataset###
def odds_calc(x):
	if x < 0:
		return - x / (100 - x)
	else:
		return 100 / (100 + x)

def profit_calc(x):
	if x < 0:
		return - 100 / x
	else:
		return x / 100

#convert the ML odds to an implied probability then scale both team's implied odds down so they sum to 100%
combo['H_ODDS'] = combo.apply(lambda x: odds_calc(x['HOME_ML']), axis = 1)
combo['A_ODDS'] = combo.apply(lambda x: odds_calc(x['AWAY_ML']), axis = 1)
combo['H_ODDS'] = combo['H_ODDS'] / (combo['H_ODDS'] + combo['A_ODDS'])
combo['A_ODDS'] = combo['A_ODDS'] / (combo['H_ODDS'] + combo['A_ODDS'])

#calculate the "advantage" for betting on a team as the ELO probability minus the implied probability
combo['H_ADV'] = combo['HOME_PROB'] - combo['H_ODDS']
combo['A_ADV'] = combo['AWAY_PROB'] - combo['A_ODDS']

#calculate an alternative "advantage" as the ELO probability's percentage difference from the implied probabiltiy
combo['H_ADV_PCT'] = combo['HOME_PROB'] / combo['H_ODDS'] - 1
combo['A_ADV_PCT'] = combo['AWAY_PROB'] / combo['A_ODDS'] - 1

#convert the ML numbers to a profit multiple and calculate how much we would have profited if we bet on one of the sides
combo['H_PROFIT'] = combo.apply(lambda x: profit_calc(x['HOME_ML']) if x['HOME_SCORE'] > x['AWAY_SCORE'] else -1, axis = 1)
combo['A_PROFIT'] = combo.apply(lambda x: profit_calc(x['AWAY_ML']) if x['AWAY_SCORE'] > x['HOME_SCORE'] else -1, axis = 1)

#save to CSV
combo.to_csv(save_destination, index = False)