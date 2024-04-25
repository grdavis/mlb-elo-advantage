import utils
import pandas as pd
from datetime import datetime, timedelta
from scraper import scrape_results_and_schedule, scrape_odds
import random
'''
MLB ELO methodology taken from https://www.baseballprospectus.com/news/article/5247/lies-damned-lies-we-are-elo/
and https://fivethirtyeight.com/features/how-our-2016-mlb-predictions-work/
'''

ADV_THRESHOLD = .04
ADV_PCT_THRESHOLD = .07 
K_FACTOR = 4
HOME_ADVANTAGE = 24
SEASON_RESET_MULT = .67 #weighting for previous end-of-season ELO, remainder of weight applied to 1500
SAVE_PATH = f"DATA/game_log_{utils.date_to_string(datetime.today())[:10]}.csv"
OUTPUT_PATH = f"OUTPUTS/{utils.date_to_string(datetime.today())[:10]} Game Predictions.csv"

class Team():
	'''
	class for storing information about a specific team including their elo rating, historical elos, and name
	'''
	def __init__(self, name, date, starting_elo):
		self.name = name
		self.elo = starting_elo

	def update_elo(self, change):
		self.elo = max(0, self.elo + change)

class ELO_Sim():
	'''
	class for keeping track of the state of a given simulation run through the data including all of the relevant Team objects for the simulation 
	'''
	def __init__(self):
		self.teams = {} #maintain a dictionary mapping string team name to a Team object
		self.date = ''

	def get_elo(self, name):
		return self.teams[name].elo

	def update_elos(self, winner, loser, delta):
		self.teams[winner].update_elo(delta)
		self.teams[loser].update_elo(-delta)

	def predict_home_winp(self, home_team, away_team):
		elo_margin = self.get_elo(home_team) - self.get_elo(away_team)
		return 1 / (1 + 10**(-elo_margin/400))

	def season_reset(self):
		for team in self.teams:
			self.teams[team].elo = self.teams[team].elo * SEASON_RESET_MULT + 1500 * (1 - SEASON_RESET_MULT)

def calc_MoV_multiplier(elo_margin, MoV):
	'''
	return the MoV multiplier based on the elo_margin and MoV
	the crazy formula here copies that from footnote 2 here: https://fivethirtyeight.com/features/how-our-2016-mlb-predictions-work/
	'''
	a = ((MoV + 1) ** .7) * 1.41
	b = (elo_margin**3*.0000000546554876) + (elo_margin**2*.00000896073139) + (elo_margin*.00244895265) + 3.4
	return a/b

def step_elo(this_sim, row, k_factor, home_adv):
	'''
	step the ELO_sim forward based on the game data provided in the specified row
	this updates the home and away team's elo ratings based on the results in row
	row: list of a neutral flag, away team, away score, home team, home score
	'''
	home, away = row['Home'], row['Away']
	homeScore, awayScore = int(row['Home_Score']), int(row['Away_Score'])
	winner, loser = home if homeScore > awayScore else away, home if awayScore > homeScore else away
	winnerScore, loserScore = homeScore if homeScore > awayScore else awayScore, homeScore if awayScore > homeScore else awayScore

	if home not in this_sim.teams: 
		this_sim.teams[home] = Team(home, this_sim.date, utils.STARTING_ELOS[home])
	if away not in this_sim.teams: 
		this_sim.teams[away] = Team(away, this_sim.date, utils.STARTING_ELOS[away])

	pre_home = this_sim.get_elo(home)
	pre_away = this_sim.get_elo(away)

	home_boost = home_adv #if row['neutral'] == 0 else 0
	Welo_0, Lelo_0 = this_sim.get_elo(winner), this_sim.get_elo(loser)
	if winner == home: Welo_0 += home_boost
	else: Lelo_0 += home_boost
	
	elo_margin = Welo_0 - Lelo_0 #winner minus loser elo
	w_winp = 1 / (1 + 10**(-elo_margin/400))
	
	MoV = winnerScore - loserScore
	MoV_multiplier = calc_MoV_multiplier(elo_margin, MoV)
	elo_delta = k_factor * MoV_multiplier * (1 - w_winp)
	this_sim.update_elos(winner, loser, elo_delta)

	post_home, post_away = this_sim.get_elo(home), this_sim.get_elo(away)
	pre_home_prob = w_winp if home == winner else 1 - w_winp
	pre_away_prob = 1 - pre_home_prob

	return pre_home, pre_away, pre_home_prob, pre_away_prob

def sim(df, k_factor, home_adv):
	'''
	Creates a new ELO_Sim object, steps it through each row in the provided dataframe.
	Stops once it reaches a game that has not occurred yet (its date >= today).
	Enriches the dataframe with 4 new columns representing the home and away elos and win
	probabilities going into the game
	'''
	output = []
	home_pre_elos, away_pre_elos = [], []
	home_pre_probs, away_pre_probs = [], []
	this_sim = ELO_Sim()
	for index, row in df.iterrows():
		#stop the sim as soon as we get to games that have not happened yet
		if utils.string_to_date(row['Date']).date() >= datetime.today().date(): break
		#apply the season reset if we have incremented by a year
		if (this_sim.date != '') and (this_sim.date[:4] != row['Date'][:4]): this_sim.season_reset()
		this_sim.date = row['Date']
		h, a, hp, ap = step_elo(this_sim, row, k_factor, home_adv)
		output.append(list(row) + [h, a, hp, ap])
	output_df = pd.DataFrame(output)
	output_df.columns = ['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML',
						'HOME_PRE_ELO', 'AWAY_PRE_ELO', 'HOME_PRE_PROB', 'AWAY_PRE_PROB']
	return this_sim, output_df

def odds_needed(winp, adv_type):
	'''
	Given an elo-predicted win probability, calculates the odds we need to see to have a certain "advantage".
	The thresholds for an advantage have been selected via backtesting. One methodology looks for an absolute
	difference in predicted and implied win probability. Another looks for a percentage difference between
	predicted and implied win probability. This function returns the odds needed to trigger each
	'''
	if adv_type == 'adv':
		original_implied = (winp - ADV_THRESHOLD)
	else:
		original_implied = (winp / (ADV_PCT_THRESHOLD + 1))

	if original_implied <= .5:
		return f"+{int(round((100 / original_implied) - 100, 0))}"
	else:
		return f"-{int(round((100 * original_implied) / (1 - original_implied), 0))}"

def predict_game(this_sim, away_team, home_team):
	winph = this_sim.predict_home_winp(home_team, away_team)
	return [away_team, home_team, round(100-winph*100, 2), round(100*winph, 2), odds_needed(1 - winph, 'adv'), 
			odds_needed(winph, 'adv'), odds_needed(1 - winph, 'adv_pct'), odds_needed(winph, 'adv_pct')]

def make_predictions(this_sim, df, pred_date = None):
	'''
	Defaults to making predictions for every game from today onwards. 
	pred_date could be used to specify a specific date to predict in format "YYYY-MM-DD"
	Output is a list of every game predicted in the format: 
	[Date, Away, Home, Away WinP, Home WinP, Away ML, Away Adv_Pct Threshold, Home ML, Home Adv_Pct Threshold]
	'''
	preds = []
	for index, row in df.iterrows():
		if pred_date == None:
			#if no prediction date specified, skip over dates prior to today
			if utils.string_to_date(row['Date']).date() < datetime.today().date(): continue
		else:
			#if prediction date specified, skip over all dates not equal to the prediction date
			if row['Date'] != pred_date: continue
		winph = this_sim.predict_home_winp(row['Home'], row['Away'])
		preds.append([row['Date'], row['Away'], row['Home'], round(100-winph*100, 2), round(100*winph, 2), 
					row['Away_ML'], odds_needed(1 - winph, 'adv_pct'), row['Home_ML'], odds_needed(winph, 'adv_pct')])
	
	output_df = pd.DataFrame(preds, columns = ["Date", "Away", "Home", "Away WinP", "Home WinP", "Away ML", "Away Threshold", "Home ML", "Home Threshold"])
	output_df.to_csv(OUTPUT_PATH)
	return output_df

def main(scrape = True, save_scrape = True, save_new_scrape = True):
	latest_filepath = utils.get_latest_data_filepath()
	games_before = latest_filepath[14:-4] #the data does not have games on or after this date
	df = pd.read_csv(latest_filepath)
	
	if scrape:
		#df columns [Date, Home, Away, Home_Score, Away_Score]
		new_df = scrape_results_and_schedule(on_or_after = games_before, save_new_scrape = save_new_scrape) 
		
		#Get a list of dates on which to scrape odds. Goes from day after last game in historical data through today
		gb_obj = utils.string_to_date(games_before)
		odds_dates = [gb_obj + timedelta(days = x) for x in range((datetime.today() - gb_obj).days + 1)]

		#Scrape odds for the days in odds_dates and merge this data with the golden schedule data
		odds_df = pd.concat([scrape_odds(utils.date_to_string(i)) for i in odds_dates]) #[Date, Home, Away, Home_Score, Away_Score, OU_Line, Home_ML, Away_ML]
		merged_df, n_matched = utils.merge_odds_and_sched(new_df, odds_df)
		print(f"Matched {n_matched} games from golden scedule with live odds")
		
		df = df.loc[df['Date'] < games_before] #cut all forward-looking rows from old df
		df = pd.concat([df, merged_df]) 
		if save_scrape: df.to_csv(SAVE_PATH, index = False)

	#get the ELO ratings up to date from the season so far
	this_sim, odf = sim(df, K_FACTOR, HOME_ADVANTAGE)

	#with the games that remain today or after, predict win probabilities
	print(make_predictions(this_sim, df, pred_date = utils.date_to_string(datetime.today())))

	return this_sim

if __name__ == '__main__':
	sim_out = main(scrape = True, save_scrape = True, save_new_scrape = False)
	print(sorted([(team, sim_out.get_elo(team)) for team in sim_out.teams], key = lambda x: x[1]))