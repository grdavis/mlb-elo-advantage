from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

DK_TEAM_MAP = {
	'KC Royals': 'KCR', 
	'DET Tigers': 'DET', 
	'OAK Athletics': 'OAK', 
	'CHI White Sox': 'CHW', 
	'ARI Diamondbacks': 'ARI', 
	'HOU Astros': 'HOU', 
	'LA Dodgers': 'LAD', 
	'BOS Red Sox': 'BOS', 
	'BAL Orioles': 'BAL', 
	'STL Cardinals': 'STL', 
	'SF Giants': 'SFG', 
	'NY Mets': 'NYM', 
	'CHI Cubs': 'CHC', 
	'SEA Mariners': 'SEA', 

	'MIA Marlins': 'FLA', 
	'PHI Phillies': 'PHI', 
	'PIT Pirates': 'PIT', 
	'NY Yankees': 'NYY', 
	'WAS Nationals': 'WSN', 
	'TOR Blue Jays': 'TOR', 
	'CIN Reds': 'CIN', 
	'CLE Guardians': 'CLE', 
	'MIL Brewers': 'MIL', 
	'TEX Rangers': 'TEX', 
	'COL Rockies': 'COL', 
	'ATL Braves': 'ATL', 
	'LA Angels': 'ANA', 
	'SD Padres': 'SDP',

	'MIN Twins': 'MIN',
	'TB Rays': 'TBD' 
}

def download_todays_odds():
	print('collecting odds data from DraftKings...')
	odds_url = "https://sportsbook.draftkings.com/leagues/baseball/mlb?category=game-lines&subcategory=game"
	chrome_options = webdriver.ChromeOptions()
	chrome_options.add_argument('--no-sandbox')
	chrome_options.add_argument('--window-size=1420,1080')
	chrome_options.add_argument('--headless')
	chrome_options.add_argument('--disable-gpu')
	driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options = chrome_options)
	
	###Option to uncomment the next five lines to run faster - does not seem to work well as part of the GitHub workflow though
	# driver.set_page_load_timeout(30)
	# try:
		# driver.get(odds_url)
	# except TimeoutException:
		# driver.execute_script("window.stop();")
	driver.get(odds_url)
	data = (driver.page_source).encode('utf-8')
	tables = BeautifulSoup(data, 'html.parser').find_all('table', {'class': 'sportsbook-table'})
	today_rows = []
	for table in tables:
		title = table.find('div', {'class': 'sportsbook-table-header__title'}).text
		if title == 'Today ': 
			today_rows.extend(table.find('tbody', {'class': 'sportsbook-table__body'}).find_all('tr'))

	if len(today_rows) == 0: 
		print('ERROR: no games today to get odds for')
		exit()

	home = False
	all_data = []
	today = datetime.now()
	for row in today_rows:
		team = DK_TEAM_MAP[row.find('div', {'class': 'event-cell__name-text'}).text]
		ml_td = row.find_all('td')[-1]
		if home:
			home_ml = ml_td.find('span', {'class': 'sportsbook-odds'})
			all_data.append(new_data + [team, home_ml.text if home_ml != None else 'NL', today.date().strftime('%Y%m%d'), today.time().strftime('%H%M%S')])
		else:
			away_ml = ml_td.find('span', {'class': 'sportsbook-odds'})
			new_data = [team, away_ml.text if away_ml != None else 'NL']
		home = not home

	df = pd.DataFrame(all_data, columns = ['AWAY', 'AWAY_ML', 'HOME', 'HOME_ML', 'PULL_DATE', 'PULL_TIMESTAMP'])
	df.to_csv('ODDS/DK_spreads_%s_%s.csv' % (today.date().strftime('%Y%m%d'), today.time().strftime('%H%M%S')), index = False)
	return df

def get_538_data():
	print('downloading 538 elo data...')
	five_url = 'https://projects.fivethirtyeight.com/mlb-api/mlb_elo_latest.csv'
	df = pd.read_csv(five_url)
	df = df.loc[df['date'] == datetime.today().strftime('%Y-%m-%d')]
	df.to_csv('ELOS/mlb_elo_latest_%s.csv' % datetime.today().strftime('%Y%m%d'))
	return df

def odds_conversion_helper(x):
	#takes in a win probability, calculates what the implied probability would need to be to take advantage,
	#and then converts that probability to the minimum odds we need to make the bet 
	x /= 1.08
	if x <= .5: return (100 - (x * 100)) / x
	return (x * -100) / (1 - x)

def create_output(dk_df, five_df):
	print('creating final output...')
	combo = five_df.merge(dk_df, how = 'inner', left_on = ['team1', 'team2'], right_on = ['HOME', 'AWAY'])
	combo['home_odds_req'] = combo['rating_prob1'].apply(odds_conversion_helper)
	combo['away_odds_req'] = combo['rating_prob2'].apply(odds_conversion_helper)
	combo['HOME_ML'] = combo['HOME_ML'].str.replace('−', '-')
	combo['AWAY_ML'] = combo['AWAY_ML'].str.replace('−', '-')
	combo = combo[['team1', 'team2', 'rating_prob1', 'rating_prob2', 'HOME_ML', 'home_odds_req', 'AWAY_ML', 'away_odds_req']]
	combo.to_csv('OUTPUTS/output_%s.csv' % datetime.today().strftime('%Y%m%d'), index = False, encoding = "utf-8")

create_output(download_todays_odds(), get_538_data())