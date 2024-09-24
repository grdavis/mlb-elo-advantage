import utils
import pandas as pd
from datetime import datetime, timedelta
from scraper import scrape_results_and_schedule, scrape_odds
import random
'''
MLB ELO methodology taken from https://www.baseballprospectus.com/news/article/5247/lies-damned-lies-we-are-elo/
and https://fivethirtyeight.com/features/how-our-2016-mlb-predictions-work/
'''

K_FACTOR = 4
PLAYOFF_K_EXTRA = 2 #gets added to K_FACTOR when game is a playoff game
PLAYOFF_MARGIN_MULT = 4/3 #elo margin multiplied by this if a playoff game
HOME_ADVANTAGE = 15 #updated from 24 on 6/7/24, updated from 17 on 7/28/24
SEASON_RESET_MULT = .67 #weighting for previous end-of-season ELO, remainder of weight applied to 1500
SAVE_PATH = f"DATA/game_log_{utils.date_to_string(datetime.today())[:10]}.csv"
SNAPSHOT_LOOKBACKS = [7, 30]

class Team():
	'''
	class for storing information about a specific team including their elo rating and name
	'''
	def __init__(self, name, date, starting_elo):
		self.name = name
		self.elo = starting_elo
		self.day_lag_snapshots = {} #maps a number of days ago to rating on that day (e.g. {7: 1500} says the team had a 1500 rating 7 days ago)
		self.division = utils.TEAM_DIVISIONS[name]
		self.league = self.division[:2]
		self.season_wins = 0
		self.season_losses = 0

	def update_elo(self, change):
		self.elo = max(0, self.elo + change)

	def snapshot(self, val):
		#add to day_lag_snapshots {val: current elo}
		self.day_lag_snapshots[val] = self.elo

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
		#In Elo system, the sum total of points is constant. Whatever points given (+delta) to the winner must come from the loser
		self.teams[winner].update_elo(delta)
		self.teams[loser].update_elo(-delta)

		#Also update the wins and losses counts for each team
		self.teams[winner].season_wins += 1
		self.teams[loser].season_losses += 1

	def predict_home_winp(self, home_team, away_team, is_playoffs):
		margin_mult = 1 if not is_playoffs else PLAYOFF_MARGIN_MULT
		elo_margin = (self.get_elo(home_team) + HOME_ADVANTAGE - self.get_elo(away_team)) * margin_mult
		return 1 / (1 + 10**(-elo_margin/400))

	def season_reset(self):
		for team in self.teams: 
			self.teams[team].elo = self.teams[team].elo * SEASON_RESET_MULT + 1500 * (1 - SEASON_RESET_MULT)
			self.teams[team].season_wins = 0
			self.teams[team].season_losses = 0

	def take_snapshots(self, val):
		#snapshot current elo for all teams - snapshot value stored in the Team object
		for team in self.teams: self.teams[team].snapshot(val)

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
	- row: dictionary with keys for 'Home', 'Away', 'Home_Score', and 'Away_Score'
	
	Also returns 4 data points in case useful: home pre-game elo, away pre-game elo, 
		home pre-game win probability, away pre-game win probability
	'''
	home, away = row['Home'], row['Away']
	homeScore, awayScore = int(row['Home_Score']), int(row['Away_Score'])
	winner = home if homeScore > awayScore else away
	loser = home if awayScore > homeScore else away
	winnerScore = homeScore if homeScore > awayScore else awayScore
	loserScore = homeScore if awayScore > homeScore else awayScore

	#instantiate teams with hard-coded Elos if we're on a team's first game in the system
	if home not in this_sim.teams: 
		this_sim.teams[home] = Team(home, this_sim.date, utils.STARTING_ELOS[home])
	if away not in this_sim.teams: 
		this_sim.teams[away] = Team(away, this_sim.date, utils.STARTING_ELOS[away])

	pre_home = this_sim.get_elo(home)
	pre_away = this_sim.get_elo(away)

	is_playoffs = True if row['Date'][5:7] in ['10', '11'] else False
	k_factor = k_factor if not is_playoffs else k_factor + PLAYOFF_K_EXTRA
	margin_mult = 1 if not is_playoffs else PLAYOFF_MARGIN_MULT

	#add the home advantage to whichever side is home
	Welo_0, Lelo_0 = this_sim.get_elo(winner), this_sim.get_elo(loser)
	if winner == home: Welo_0 += home_adv
	else: Lelo_0 += home_adv
	
	elo_margin = (Welo_0 - Lelo_0) * margin_mult #winner minus loser elo multipled by extra margin if playoffs
	w_winp = 1 / (1 + 10**(-elo_margin/400))
	
	MoV = winnerScore - loserScore #how much did winner win by
	MoV_multiplier = calc_MoV_multiplier(elo_margin, MoV)
	elo_delta = k_factor * MoV_multiplier * (1 - w_winp)
	this_sim.update_elos(winner, loser, elo_delta)

	#prepare to return
	pre_home_prob = w_winp if home == winner else 1 - w_winp
	pre_away_prob = 1 - pre_home_prob

	return pre_home, pre_away, pre_home_prob, pre_away_prob

def sim(df, k_factor, home_adv, snapshots = SNAPSHOT_LOOKBACKS):
	'''
	Creates a new ELO_Sim object, steps it through each row in the provided dataframe.
	Stops once it reaches a game that has not occurred yet (its date >= today).
	Returns the sim object and an enriched dataframe with 4 new columns representing 
	the home and away elos and win probabilities going into the game
	'''
	output = []
	this_sim = ELO_Sim()
	bdate = None
	today_date = utils.today_date_string()
	snapshot_dates = {utils.shift_dstring(today_date, -i) : i for i in snapshots}
	snapshot_day = False
	
	for index, row in df.iterrows():
		#if we get to a bdate on which we should take a snapshot, set snapshot_day = True
		if bdate in snapshot_dates: snapshot_day = True

		#if this new row does not match the existing bdate, we've finished the previous bdate
		#if that bdate was a snapshot_day, take a snapshot, and set snapshot_day = False
		if bdate != row['Date'] and snapshot_day: 
			this_sim.take_snapshots(snapshot_dates[bdate])
			snapshot_day = False

		bdate = row['Date']
		
		#stop the sim as soon as we get to games that have not happened yet
		if bdate >= today_date: break
		
		#apply the season reset and snapshot if we have incremented by a year
		if (this_sim.date != '') and (this_sim.date[:4] != bdate[:4]): 
			this_sim.season_reset()
			this_sim.take_snapshots('pre-season')
		
		#step the Elo system forward based on the results in the row and enrich the output data
		this_sim.date = bdate
		h, a, hp, ap = step_elo(this_sim, row, k_factor, home_adv)
		output.append(list(row) + [h, a, hp, ap])

	output_df = pd.DataFrame(output)
	output_df.columns = ['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML',
						'HOME_PRE_ELO', 'AWAY_PRE_ELO', 'HOME_PRE_PROB', 'AWAY_PRE_PROB']
	
	#bring together the output_df which reflects historical games with the remaining future games
	sched_df = df.loc[df['Date'] >= bdate]
	combo = pd.concat([output_df, sched_df], axis = 0, join = 'outer')
	
	return this_sim, combo

def main(scrape = True, save_scrape = True, save_new_scrape = True, print_ratings = False):
	'''
	By default this method scrapes the latest data up to today that is not already in our dataset, 
	saves the new data, saves the combined old + new data, and returns the sim object and a dataframe of the old + new
	It can also optionally print out the team's Elo ratings sorted from greatest to least
	'''
	latest_filepath = utils.get_latest_data_filepath()
	games_before = latest_filepath[14:-4] #the data does not have games on or after this date
	df = pd.read_csv(latest_filepath)
	
	#if the latest data is not up to date and scrape is true, continue with the scraping
	if utils.string_to_date(games_before).date() < datetime.today().date() and scrape:
		#grabs the latest "golden" source of truth schedule
		new_df = scrape_results_and_schedule(on_or_after = games_before, save_new_scrape = save_new_scrape) 

		#Get a list of dates on which to scrape odds - goes from day after last game in historical data through today
		#Scrape odds for the days in odds_dates and merge this data with the golden schedule data
		gb_obj = utils.string_to_date(games_before)
		odds_dates = [gb_obj + timedelta(days = x) for x in range((datetime.today() - gb_obj).days + 1)]
		odds_df = pd.concat([scrape_odds(utils.date_to_string(i)) for i in odds_dates]) 
		merged_df, n_matched = utils.merge_odds_and_sched(new_df, odds_df)
		print(f"Matched {n_matched} games from golden scedule with live odds")
		
		#combine the merged golden schedule + odds data with the historical, existing data
		df = df.loc[df['Date'] < games_before] #cut all forward-looking rows from old df
		df = pd.concat([df, merged_df]) 
		if save_scrape: df.to_csv(SAVE_PATH, index = False)

	#run the elo sim on the assembled DataFrame
	sim_out, odf = sim(df, K_FACTOR, HOME_ADVANTAGE)
	if print_ratings: 
		print(f'Ratings based on games before {games_before}...')
		print(sorted([(team, sim_out.get_elo(team)) for team in sim_out.teams], key = lambda x: x[1]))

	return sim_out, odf

if __name__ == '__main__':
	main(scrape = False, save_scrape = False, save_new_scrape = False, print_ratings = True)