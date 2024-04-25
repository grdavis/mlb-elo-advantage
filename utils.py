import re
import os
from datetime import datetime

DATA_FOLDER = 'DATA/'

#ELOs to start the 2023 season - This is when the ELO model restarts every run
STARTING_ELOS = { 
	'NYY': 1543.060843,
	'SFG': 1514.395798,
	'WSN': 1454.556936,
	'ATL': 1548.169577,
	'BOS': 1498.037778,
	'BAL': 1498.610307,
	'CHC': 1489.04351,
	'MIL': 1512.200668,
	'TBD': 1523.683927,
	'DET': 1473.797327,
	'TEX': 1476.160323,
	'PHI': 1531.204297,
	'CIN': 1462.909072,
	'PIT': 1457.555517,
	'FLA': 1477.833312,
	'NYM': 1528.751503,
	'KCR': 1470.669424,
	'MIN': 1495.072693,
	'STL': 1524.159769,
	'TOR': 1531.601034,
	'HOU': 1569.837664,
	'CHW': 1499.436735,
	'SDP': 1521.984559,
	'COL': 1471.699602,
	'OAK': 1469.00074,
	'ANA': 1492.615718,
	'LAD': 1569.317852,
	'ARI': 1494.078935,
	'SEA': 1524.991153,
	'CLE': 1525.558497
}

BR_TEAM_MAP = {
	'New York Yankees': 'NYY',
	'San Francisco Giants': 'SFG',
	'Washington Nationals': 'WSN',
	'Atlanta Braves': 'ATL',
	'Boston Red Sox': 'BOS',
	'Baltimore Orioles': 'BAL',
	'Chicago Cubs': 'CHC',
	'Milwaukee Brewers': 'MIL',
	'Tampa Bay Rays': 'TBD',
	'Detroit Tigers': 'DET',
	'Texas Rangers': 'TEX',
	'Philadelphia Phillies': 'PHI',
	'Cincinnati Reds': 'CIN',
	'Pittsburgh Pirates': 'PIT',
	'Miami Marlins': 'FLA',
	'New York Mets': 'NYM',
	'Kansas City Royals': 'KCR',
	'Minnesota Twins': 'MIN',
	'St. Louis Cardinals': 'STL',
	'Toronto Blue Jays': 'TOR',
	'Houston Astros': 'HOU',
	'Chicago White Sox': 'CHW',
	'San Diego Padres': 'SDP',
	'Colorado Rockies': 'COL',
	'Oakland Athletics': 'OAK',
	'Los Angeles Angels': 'ANA',
	'Los Angeles Dodgers': 'LAD',
	"Arizona D'Backs": 'ARI',
	'Seattle Mariners': 'SEA',
	'Cleveland Guardians': 'CLE'
}

SO_TEAM_MAP = {
	'Yankees': 'NYY',
	'Giants': 'SFG',
	'Nationals': 'WSN',
	'Braves': 'ATL',
	'Red Sox': 'BOS',
	'Orioles': 'BAL',
	'Cubs': 'CHC',
	'Brewers': 'MIL',
	'Rays': 'TBD',
	'Tigers': 'DET',
	'Rangers': 'TEX',
	'Phillies': 'PHI',
	'Reds': 'CIN',
	'Pirates': 'PIT',
	'Marlins': 'FLA',
	'Mets': 'NYM',
	'Royals': 'KCR',
	'Twins': 'MIN',
	'Cardinals': 'STL',
	'Blue Jays': 'TOR',
	'Astros': 'HOU',
	'White Sox': 'CHW',
	'Padres': 'SDP',
	'Rockies': 'COL',
	'Athletics': 'OAK',
	'Angels': 'ANA',
	'Dodgers': 'LAD',
	"Diamondbacks": 'ARI',
	'Mariners': 'SEA',
	'Guardians': 'CLE'
}

def date_to_string(obj): return obj.strftime('%Y-%m-%d')

def string_to_date(date_str): return datetime.strptime(date_str, '%Y-%m-%d')

def merge_odds_and_sched(sched, odds):
	'''
	Take in one dataframe containing the golden schedule and another containing a schedule + odds
	Merge these dataframes to create a golden schedule enriched with current odds where matches could be found
	Outputs a df ['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML']
	'''
	#add a column to each df to break ties in the case of doubleheaders on the same day
	sched['matchup_on_date'] = sched.groupby(['Date', 'Home', 'Away']).transform('cumcount') + 1
	odds['matchup_on_date'] = odds.groupby(['Date', 'Home', 'Away']).transform('cumcount') + 1
	sched = sched.merge(odds, how = 'left', on = ['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'matchup_on_date'])
	matched = sched.loc[~sched['OU_Line'].isna()] #roundabout way of counting how many we matched from scheduled and odds
	return sched[['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML']], matched.shape[0]

def get_latest_data_filepath():
	'''
	searches the data folder and returns the most recent data
	'''
	r = re.compile("game_log_.*.csv")
	elligible_data = list(filter(r.match, os.listdir(DATA_FOLDER)))
	return DATA_FOLDER + sorted(elligible_data, key = lambda x: x[9:19], reverse = True)[0]