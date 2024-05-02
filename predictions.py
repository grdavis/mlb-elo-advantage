import elo
import utils
from datetime import datetime
import pandas as pd

OUTPUT_PATH = f"OUTPUTS/{utils.date_to_string(datetime.today())[:10]} Game Predictions.csv"

def odds_needed(winp, adv_type):
	'''
	Given an elo-predicted win probability, calculates the odds we need to see to have a certain "advantage".
	The thresholds for an advantage have been selected via backtesting. One methodology looks for an absolute
	difference in predicted and implied win probability. Another looks for a percentage difference between
	predicted and implied win probability. This function returns the odds needed to trigger based on adv_type
	- adv_type: should be a string 'adv' to indicate we want to use the absolute difference method
	'''
	if adv_type == 'adv':
		original_implied = (winp - ADV_THRESHOLD)
	else:
		original_implied = (winp / (ADV_PCT_THRESHOLD + 1))

	if original_implied <= .5:
		return f"+{int(round((100 / original_implied) - 100, 0))}"
	else:
		return f"-{int(round((100 * original_implied) / (1 - original_implied), 0))}"

def make_predictions(this_sim, df, pred_date = None):
	'''
	Defaults to making predictions for every game from today onwards. 
	pred_date could be used to specify a specific date to predict in format "YYYY-MM-DD"
	Output is a list of every game predicted in the format: 
	[Date, Away, Home, Away WinP, Home WinP, Away ML, Away Adv_Pct Threshold, Home ML, Home Adv_Pct Threshold]
	Output presented as a plotly table and saved to markdown for presentation on GitHub pages
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
	utils.table_output(output_df, 'Game Predictions Based on Ratings through ' + this_sim.date)
	
	#save the predictions output and ratings in markdown where github pages can find it
	ratings = pd.DataFrame(sorted([(team, this_sim.get_elo(team)) for team in this_sim.teams], key = lambda x: x[1], reverse = True), columns = ['Team', 'Elo Rating']) 
	utils.save_markdown_df(output_df, ratings, pred_date)

def main():
	'''
	This method leverages elo.py to get the latest old data, scrape to update, and get the odds for upcoming games
	Using that updated dataframe, this script predicts games today by default
	'''
	this_sim, df = elo.main()
	make_predictions(this_sim, df, pred_date = utils.date_to_string(datetime.today()))

if __name__ == '__main__':
	main()