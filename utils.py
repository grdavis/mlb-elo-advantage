import re
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

DATA_FOLDER = 'DATA/'
DOCS_FOLDER = 'docs/'
OUTPUTS_FOLDER = 'OUTPUTS/'
UNIT = 1

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
	'Athletics': 'OAK',
	'Los Angeles Angels': 'ANA',
	'Los Angeles Dodgers': 'LAD',
	"Arizona D'Backs": 'ARI',
	'Seattle Mariners': 'SEA',
	'Cleveland Guardians': 'CLE',
	'Cleveland Indians': 'CLE',
	'Oakland Athletics': 'OAK'
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

TEAM_DIVISIONS = {
	'NYY': 'AL East',
	'SFG': 'NL West',
	'WSN': 'NL East',
	'ATL': 'NL East',
	'BOS': 'AL East',
	'BAL': 'AL East',
	'CHC': 'NL Central',
	'MIL': 'NL Central',
	'TBD': 'AL East',
	'DET': 'AL Central',
	'TEX': 'AL West',
	'PHI': 'NL East',
	'CIN': 'NL Central',
	'PIT': 'NL Central',
	'FLA': 'NL East',
	'NYM': 'NL East',
	'KCR': 'AL Central',
	'MIN': 'AL Central',
	'STL': 'NL Central',
	'TOR': 'AL East',
	'HOU': 'AL West',
	'CHW': 'AL Central',
	'SDP': 'NL West',
	'COL': 'NL West',
	'OAK': 'AL West',
	'ANA': 'AL West',
	'LAD': 'NL West',
	'ARI': 'NL West',
	'SEA': 'AL West',
	'CLE': 'AL Central'
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
	
	NOTE: This is a LEFT JOIN - we keep ALL games from the golden schedule (sched) and enrich with odds where available.
	Games without odds will have NaN/empty values for OU_Line, Home_ML, Away_ML.
	'''
	#add a column to each df to break ties in the case of doubleheaders on the same day
	sched['matchup_on_date'] = sched.groupby(['Date', 'Home', 'Away']).cumcount() + 1
	
	# Only add matchup_on_date to odds if it's not empty
	if not odds.empty:
		odds['matchup_on_date'] = odds.groupby(['Date', 'Home', 'Away']).cumcount() + 1
		
		# Merge on Date, Home, Away, and matchup_on_date only (NOT on scores)
		# This prevents score format mismatches from breaking the merge
		sched = sched.merge(odds[['Date', 'Home', 'Away', 'matchup_on_date', 'OU_Line', 'Home_ML', 'Away_ML']], 
							how = 'left', 
							on = ['Date', 'Home', 'Away', 'matchup_on_date'])
	else:
		# If odds is empty, just add empty columns
		sched['OU_Line'] = None
		sched['Home_ML'] = None
		sched['Away_ML'] = None
	
	matched = sched.loc[~sched['OU_Line'].isna()] #roundabout way of counting how many we matched from scheduled and odds
	return sched[['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML']], matched.shape[0]

def get_latest_data_filepath():
	'''
	searches the data folder and returns the most recent data
	'''
	r = re.compile("game_log_.*.csv")
	elligible_data = list(filter(r.match, os.listdir(DATA_FOLDER)))
	return DATA_FOLDER + sorted(elligible_data, key = lambda x: x[9:19], reverse = True)[0]

def save_markdown_df(predictions, ratings, date_str, performance, sims = 50000):
	'''
	Takes in a predictions dataframe of today's predictions and a table with the team rankings
	Converts tables to markdown, and saves them in the same file in the docs folder for GitHub pages to find
	Also takes in the recent performance of the betting recommendations and publishes performance
	'''
	r7, b7 = performance[0]
	r30, r365 = performance[1][0], performance[2][0]
	with open(f"{DOCS_FOLDER}/index.md", 'w') as md:
		md.write(f'# MLB Elo Game Predictions and Playoff Probabilities for {date_str} - @grdavis\n')
		md.write("Below are predictions for today's MLB games using an ELO rating methodology. Check out the full [mlb-elo-advantage](https://github.com/grdavis/mlb-elo-advantage) repository on github to see methodology and more.\n\n")
		md.write("The thresholds indicate at what odds the model thinks there is value in betting on a team. These thresholds were selected via backtesting over the last 2 years (2.7K games). ")
		md.write(f"For transparency, these recommendations have been triggered for {b7}% of games and have a {r7}% ROI over the last 7 days. ROI is {r30}% over the last 30 days and {r365}% over the last 365.\n\n")
		predictions.to_markdown(buf = md, index = False)
		md.write('\n\n')
		md.write('# Team Elo Ratings\n')
		md.write(f"This table summarizes each team's Elo rating and their chances of making it to various stages of the postseason based on {sims} simulations of the rest of the regular season and playoffs\n\n")
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
	#wager enough to win a UNIT on favorites; wager a UNIT on underdogs
	if x < 0: 
		return - UNIT * x / 100
	else: 
		return UNIT

def profit_calc(x, wager):
	#x is the ML odds; wager is the amount wagered on those odds
	#return the amount of profit earned from this wager on those odds
    if x < 0:
        return - 100 / x * wager
    else:
        return x / 100 * wager