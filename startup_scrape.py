from bs4 import BeautifulSoup
import requests
import csv
from tqdm import tqdm

DESTINATION = 'game_logs_2019_to_20230605.csv'
URL = 'https://www.oddsshark.com/stats/gamelog/baseball/mlb/NUM?season=SEA'
SEASONS = [2019, 2020, 2021, 2022, 2023]
TEAM_MAP = {
	27018: 'CHW',
	27024: 'CLE',
	26999: 'DET',
	27006: 'KCR',
	27005: 'MIN',
	
	27008: 'BAL',
	27021: 'BOS',
	27001: 'NYY',
	27003: 'TBD',
	27010: 'TOR',

	27023: 'HOU',
	26998: 'ANA',
	27016: 'OAK',
	27011: 'SEA',
	27002: 'TEX',

	27020: 'CHC',
	27000: 'CIN',
	27012: 'MIL',
	27013: 'PIT',
	27019: 'STL',

	27009: 'ATL',
	27022: 'FLA',
	27014: 'NYM',
	26995: 'PHI',
	27017: 'WSN',

	27007: 'ARI',
	27004: 'COL',
	27015: 'LAD',
	26996: 'SDP',
	26997: 'SFG',
}

def save_data(filepath, data):
	with open(filepath, "w") as f:
		wr = csv.writer(f)
		for row in data:
			wr.writerow(row)

all_data = []
for team in tqdm(TEAM_MAP):
	for season in SEASONS:
		turl = URL.replace('NUM', str(team)).replace('SEA', str(season))
		data = requests.get(turl).content
		table = BeautifulSoup(data,'html.parser').find('tbody')
		rows = table.find_all("tr")
		for row in rows:
			all_data.append([season, TEAM_MAP[team]] + [i.text for i in row.find_all('td')])

save_data(DESTINATION, all_data)




















