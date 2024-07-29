import pandas as pd
import re
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

DATA_FOLDER = 'DATA/'
DOCS_FOLDER = 'docs/'
OUTPUTS_FOLDER = 'OUTPUTS/'
UNIT = 1

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

def today_date_string(): return date_to_string(datetime.today())

def string_to_date(date_str): return datetime.strptime(date_str, '%Y-%m-%d')

def shift_dstring(day_string, days):
	#take in a string in the format YYYYMMDD and return it shifted the specified number of days (could be positive or negative days)
	return date_to_string(string_to_date(day_string) + timedelta(days = days))

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

def save_markdown_df(predictions, ratings, date_str, performance):
	'''
	Takes in a predictions dataframe of today's predictions and a table with the team rankings
	Converts tables to markdown, and saves them in the same file in the docs folder for GitHub pages to find
	Also takes in the recent performance of the betting recommendations and publishes performance
	'''
	r7, b7 = performance[0]
	r30, r365 = performance[1][0], performance[2][0]
	with open(f"{DOCS_FOLDER}/index.md", 'w') as md:
		md.write(f'# MLB Elo Game Predictions for {date_str} - @grdavis\n')
		md.write("Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.\n\n")
		md.write("The thresholds indicate at what odds the model thinks there is value in betting on a team. These thresholds were selected via backtesting since the start of the 2023 season. ")
		md.write(f"For transparency, these recommendations have been triggered for {b7}% of games and have a {r7}% ROI over the last 7 days. ROI is {r30}% over the last 30 days and {r365}% over the last 365.\n\n")
		predictions.to_markdown(buf = md, index = False)
		md.write('\n\n')
		md.write('# Team Elo Ratings\n')
		ratings.index = ratings.index + 1
		ratings.to_markdown(buf = md, index = True)

def remove_files(to_remove, k):
	'''
	Deletes all but the first k files provided in to_remove. Assuming to_remove is sorted chronologically,
	this removes all but the most recent k files
	'''
	if len(to_remove) > k:
		for f in to_remove[k:]:
			os.remove(f)

def clean_up_old_outputs_and_data():
	'''
	Goes through the outputs and data folders to remove all but the 3 latest files. The files are
	really only saved for debugging purposes anyways, so no need to have so much extra data.
	'''
	#DATA
	r = re.compile("game_log_.*.csv")
	elligible_data = list(filter(r.match, os.listdir(DATA_FOLDER)))
	sorted_files = sorted(elligible_data, key = lambda x: x[9:19], reverse = True)
	remove_files([DATA_FOLDER + f for f in sorted_files], 3)

	#OUTPUTS
	r = re.compile(".*Game Predictions.*.csv")
	elligible_data = list(filter(r.match, os.listdir(OUTPUTS_FOLDER)))
	sorted_files = sorted(elligible_data, key = lambda x: x[-14:-4], reverse = True)
	remove_files([OUTPUTS_FOLDER + f for f in sorted_files], 3)

def table_output(df, table_title, order = None):
	'''
	saves the specified dataframe as a csv and outputs it in the form of a Plotly table
	df: dataframe to structure in the form of a plotly table for .html output
	table_title: title used in table
	order: optional list of strings that specifies an order the columns should be presented in
	'''
	if order != None:
		df = df[order]
	df.to_csv(OUTPUTS_FOLDER + table_title + '.csv', index = False)
	fig = go.Figure(data=[go.Table(
	    header=dict(values=list(df.columns),
	                fill_color='paleturquoise',
	                align='left'),
	    cells=dict(values=[df[col].to_list() for col in list(df)],
	               # fill_color='lavender',
	               align='left'))
	])
	fig.update_layout(title = {'text': table_title, 'xanchor': 'center', 'x': .5})
	fig.show()

def odds_calc(x):
    if x < 0:
        return - x / (100 - x)
    else:
        return 100 / (100 + x)

def wager_calc(x):
    #wager either 1 UNIT on the underdog or enough to win 1 UNIT on the favorite
    if x < 0:
        return - UNIT * x / 100
    else:
        return UNIT

def profit_calc(x):
    #wager either 1 UNIT on the underdog or enough to win 1 UNIT on the favorite
    if x < 0:
        return UNIT
    else:
        return x / 100 * UNIT
								
def assemble_results_and_predictions():
    #grab the ratings and predictions for every historical game then enrich the dataset
    latest_df = pd.read_csv(get_latest_data_filepath())
    latest_sim, combo = sim(latest_df, K_FACTOR, HOME_ADVANTAGE)
    combo = combo.loc[combo['Date'] < today_date_string()]

    #convert the ML odds to an implied probability, they will sum to more than 100% because of the book's vig
    combo['H_ODDS'] = combo.apply(lambda x: odds_calc(x['Home_ML']), axis = 1)
    combo['A_ODDS'] = combo.apply(lambda x: odds_calc(x['Away_ML']), axis = 1)

    #calculate the "advantage" for betting on a team as the ELO probability minus the implied probability
    combo['H_ADV'] = combo['HOME_PRE_PROB'] - combo['H_ODDS']
    combo['A_ADV'] = combo['AWAY_PRE_PROB'] - combo['A_ODDS']

    #calculate an alternative "advantage" as the ELO probability's percentage difference from the implied probabiltiy
    combo['H_ADV_PCT'] = combo['HOME_PRE_PROB'] / combo['H_ODDS'] - 1
    combo['A_ADV_PCT'] = combo['AWAY_PRE_PROB'] / combo['A_ODDS'] - 1

    #convert the ML numbers to a profit multiple and calculate how much we would have profited if we bet on one of the sides
    combo['H_WAGER'] = combo.apply(lambda x: wager_calc(x['Home_ML']), axis = 1)
    combo['A_WAGER'] = combo.apply(lambda x: wager_calc(x['Away_ML']), axis = 1)
    combo['H_PROFIT'] = combo.apply(lambda x: profit_calc(x['Home_ML']) if int(x['Home_Score']) > int(x['Away_Score']) else -x['H_WAGER'], axis = 1)
    combo['A_PROFIT'] = combo.apply(lambda x: profit_calc(x['Away_ML']) if int(x['Away_Score']) > int(x['Home_Score']) else -x['A_WAGER'], axis = 1)

    return combo

def convert_to_betting_rows(combo, adv_to_use):
    #convert dataframe from one row per game to one row per possible betting side (2 rows per game)
    mdf = pd.melt(combo, id_vars = ['H_WAGER', 'A_WAGER', 'H_PROFIT', 'A_PROFIT'], value_vars = [f'H_{adv_to_use}', f'A_{adv_to_use}'])
    mdf['WAGER'] = mdf.apply(lambda x: x['H_WAGER'] if x['variable'] == f'H_{adv_to_use}' else x['A_WAGER'], axis = 1)
    mdf['PROFIT'] = mdf.apply(lambda x: x['H_PROFIT'] if x['variable'] == f'H_{adv_to_use}' else x['A_PROFIT'], axis = 1)
    mdf.rename(columns = {'value': 'ADVANTAGE'}, inplace = True)
    return mdf

def eval_recent_performance(recent_days, adv_to_use, threshold):
    #evaluate performance of betting thresholds over recent days
    combo = assemble_results_and_predictions()
    combo = combo.loc[combo['Date'] >= shift_dstring(today_date_string(), -recent_days)]
    
    mdf = convert_to_betting_rows(combo, adv_to_use)
    total_games = mdf.shape[0] / 2 #divide by two since mdf has one row for each side of the game

    #return the ROI and bet rate with current threshold used for predictions
    aggs = mdf.loc[mdf['ADVANTAGE'] >= threshold][['WAGER', 'PROFIT']].agg(['size', 'sum'])
    return (round(aggs.loc['sum', 'PROFIT'] / aggs.loc['sum', 'WAGER'] * 100, 2), round(aggs.loc['size', 'WAGER'] / total_games * 100))