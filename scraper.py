from bs4 import BeautifulSoup
import requests
import csv
from datetime import datetime, timedelta
import pandas as pd
from utils import BR_TEAM_MAP, SO_TEAM_MAP, date_to_string, string_to_date

SEASON = 2025
SCHEDULE_URL = f'https://www.baseball-reference.com/leagues/majors/{SEASON}-schedule.shtml'
ODDS_URL = 'https://www.scoresandodds.com/mlb?date=YEAR-MONTH-DAY'
SAVE_PATH = f"DATA/new_game_log_{date_to_string(datetime.today())[:10]}.csv"

def get_info_from_final_row(row):
	#collect the teams, scores, and MLs for a game that has already finished
	team = row.find('span', {'class': 'team-name'})
	team = team.find('a').text.strip()

	team_score = row.find('td', {'class': 'event-card-score'})
	team_score = '' if team_score == None else team_score.text.strip()

	team_ml = row.find('td', {'data-field': 'live-moneyline'})
	if team_ml == None: 
		team_ml = 'NA'
	else:
		team_ml = team_ml.find('span', {'class': 'data-value'})
		if team_ml == None:
			team_ml = 'NA' 
		elif team_ml.text.strip(" +") == 'even':
			team_ml = 100
		else:
			team_ml = int(team_ml.text.strip(" +"))

	ou = row.find('td', {'data-field': 'live-total'})
	if ou == None:
		ou = 'NA'
	else:
		ou = ou.find('span', {'class': 'data-value'})
		ou = 'NA' if ou == None else ou.text.strip()[1:]

	return team, team_score, team_ml, ou

def get_info_from_scheduled_row(row):
	#collect the teams and MLs for a game that has not finished
	team = row.find('span', {'class': 'team-name'})
	team = team.find('a').text.strip()
	
	team_ml = row.find('td', {'data-field': 'current-moneyline'})
	if team_ml == None:
		team_ml = 'NA'
	else:
		team_ml.find('span', {'class': 'data-value'})
		if team_ml == None:
			team_ml = 'NA' 
		elif team_ml.text.strip(" +") == 'even':
			team_ml = 100
		else:
			team_ml = int(team_ml.text.strip(" +"))
	
	ou = row.find('td', {'data-field': 'current-total'})
	if ou == None:
		ou = 'NA'
	else:
		ou = ou.find('span', {'class': 'data-value'})
		ou = 'NA' if ou == None else ou.text.strip()[1:]
	
	return team, team_ml, ou

def scrape_odds(date_str):
	'''
	Visits scoresandodds.com looking for games scheduled on the date_obj day
	For every game on that day, grabs the team names away and home MLs (if present),
	O/U line, and scores (if present)
	'''
	print(f'Scraping odds for {date_str}...')
	date_obj = string_to_date(date_str)
	scores_final = date_obj.date() < datetime.today().date() #T/F flag figuring out if these games already happened
	day, month, year = str(date_obj.day), str(date_obj.month), str(date_obj.year)
	url = ODDS_URL.replace("DAY", day).replace("MONTH", month).replace("YEAR", year)

	response = requests.get(url)
	if response.status_code != 200:
		print(f'ERROR: Response Code {response.status_code}')
	data = response.content
	table_divs = BeautifulSoup(data, 'html.parser').find_all('tbody')
	
	day_stats = []
	for game in table_divs:
		away_row, home_row = game.find_all('tr')
		if scores_final:
			away, away_score, away_ml, aou = get_info_from_final_row(away_row)
			home, home_score, home_ml, hou = get_info_from_final_row(home_row)
			if away_score == '' or home_score == '': continue #skip games that did not start this day (postponed)
			day_stats.append([date_to_string(date_obj), SO_TEAM_MAP[home], SO_TEAM_MAP[away], home_score, away_score, hou, home_ml, away_ml])
		else:
			away, away_ml, aou = get_info_from_scheduled_row(away_row)
			home, home_ml, hou = get_info_from_scheduled_row(home_row)
			day_stats.append([date_to_string(date_obj), SO_TEAM_MAP[home], SO_TEAM_MAP[away], '', '', hou, home_ml, away_ml])

	return pd.DataFrame(day_stats, columns = ['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML'])

def scrape_results_and_schedule(on_or_after, save_new_scrape = True):
	'''
	Visits Baseball Reference season schedule page and grabs all scores on_or_after a date.
	on_or_after formatted as 'YYYYMMDD'. Output is a DataFrame
	'''
	print(f'Scraping schedule for games on or after {on_or_after}...')
	all_data = []
	data = requests.get(SCHEDULE_URL).content
	section_content = BeautifulSoup(data,'html.parser').find_all('div', {'class': 'section_content'})
	#there should be one section_content for regular season and one for post-season
	for sc in section_content:
		sections = sc.find_all("div")
		for section in sections:
			date = section.find('h3').text
			if date == "Today's Games":
				date = date_to_string(datetime.today())
			else:
				date = date_to_string(datetime.strptime(date[date.find(',')+2:], '%B %d, %Y'))
			
			#don't bother re-collecting data we already have
			if date < on_or_after: continue
			
			rows = section.find_all("p")
			for row in rows:
				this_row_contents = row.text
				this_row_contents = this_row_contents.split('\n')[1:-2]
				if this_row_contents == [] or this_row_contents[1] == '(Spring)': continue
				is_playoffs = date >= '2025-09-29'
				if len(this_row_contents) in [6, 7]: 
					#future games, remove the postseason series designation for easier scraping if playoffs
					if is_playoffs: this_row_contents.pop(1)
					all_data.append([date,
						BR_TEAM_MAP[this_row_contents[4].strip(' ')],
						BR_TEAM_MAP[this_row_contents[1]],
						'', ''])
				elif len(this_row_contents) == 5:
					#games already occurred
					all_data.append([date,
						BR_TEAM_MAP[this_row_contents[3].strip(' ')], 
						BR_TEAM_MAP[this_row_contents[0].strip(' ')], 
						this_row_contents[4].strip(' ').strip('(').strip(')'),
						this_row_contents[1].strip(' ').strip('(').strip(')')
						])

	df = pd.DataFrame(all_data, columns = ['Date', 'Home', 'Away', 'Home_Score', 'Away_Score'])
	if save_new_scrape: df.to_csv(SAVE_PATH, index = False)
	return df

# scrape_results_and_schedule('2024-11-06', save_new_scrape = False)
# print(scrape_odds('2024-06-29'))