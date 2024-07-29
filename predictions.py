import elo
import utils
from datetime import datetime
import pandas as pd

OUTPUT_PATH = f"OUTPUTS/{utils.date_to_string(datetime.today())[:10]} Game Predictions.csv"
ADV_THRESHOLD = .04
ADV_PCT_THRESHOLD = .11 #changed from .08 on 7/28/24 after more tuning adjustments to home advantage 
ADV_TO_USE = 'ADV_PCT'

def assemble_results_and_predictions():
    #grab the ratings and predictions for every historical game then enrich the dataset
    latest_df = pd.read_csv(utils.get_latest_data_filepath())
    latest_sim, combo = elo.sim(latest_df, elo.K_FACTOR, elo.HOME_ADVANTAGE)
    combo = combo.loc[combo['Date'] < utils.today_date_string()]

    #convert the ML odds to an implied probability, they will sum to more than 100% because of the book's vig
    combo['H_ODDS'] = combo.apply(lambda x: utils.odds_calc(x['Home_ML']), axis = 1)
    combo['A_ODDS'] = combo.apply(lambda x: utils.odds_calc(x['Away_ML']), axis = 1)

    #calculate the "advantage" for betting on a team as the ELO probability minus the implied probability
    combo['H_ADV'] = combo['HOME_PRE_PROB'] - combo['H_ODDS']
    combo['A_ADV'] = combo['AWAY_PRE_PROB'] - combo['A_ODDS']

    #calculate an alternative "advantage" as the ELO probability's percentage difference from the implied probabiltiy
    combo['H_ADV_PCT'] = combo['HOME_PRE_PROB'] / combo['H_ODDS'] - 1
    combo['A_ADV_PCT'] = combo['AWAY_PRE_PROB'] / combo['A_ODDS'] - 1

    #convert the ML numbers to a profit multiple and calculate how much we would have profited if we bet on one of the sides
    combo['H_WAGER'] = combo.apply(lambda x: utils.wager_calc(x['Home_ML']), axis = 1)
    combo['A_WAGER'] = combo.apply(lambda x: utils.wager_calc(x['Away_ML']), axis = 1)
    combo['H_PROFIT'] = combo.apply(lambda x: utils.profit_calc(x['Home_ML']) if int(x['Home_Score']) > int(x['Away_Score']) else -x['H_WAGER'], axis = 1)
    combo['A_PROFIT'] = combo.apply(lambda x: utils.profit_calc(x['Away_ML']) if int(x['Away_Score']) > int(x['Home_Score']) else -x['A_WAGER'], axis = 1)

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
    combo = combo.loc[combo['Date'] >= utils.shift_dstring(today_date_string(), -recent_days)]
    
    mdf = convert_to_betting_rows(combo, adv_to_use)
    total_games = mdf.shape[0] / 2 #divide by two since mdf has one row for each side of the game

    #return the ROI and bet rate with current threshold used for predictions
    aggs = mdf.loc[mdf['ADVANTAGE'] >= threshold][['WAGER', 'PROFIT']].agg(['size', 'sum'])
    return (round(aggs.loc['sum', 'PROFIT'] / aggs.loc['sum', 'WAGER'] * 100, 2), round(aggs.loc['size', 'WAGER'] / total_games * 100))

def odds_needed(winp, adv_type):
	'''
	Given an elo-predicted win probability, calculates the odds we need to see to have a certain "advantage".
	The thresholds for an advantage have been selected via backtesting. One methodology looks for an absolute
	difference in predicted and implied win probability. Another looks for a percentage difference between
	predicted and implied win probability. This function returns the odds needed to trigger based on adv_type
	- adv_type: should be a string 'ADV' to indicate we want to use the absolute difference method 
				it could be anything else to indicate we want to use the percentage difference method
	'''
	if adv_type == 'ADV':
		original_implied = (winp - ADV_THRESHOLD)
	else:
		original_implied = (winp / (ADV_PCT_THRESHOLD + 1))

	if original_implied <= .5:
		return f"+{int(round((100 / original_implied) - 100, 0))}"
	else:
		return f"-{int(round((100 * original_implied) / (1 - original_implied), 0))}"

def clean_up_ratings(this_sim):
	out_data = []
	for team in this_sim.teams:
		this_row = [team, round(this_sim.get_elo(team))]
		snaps = [this_sim.teams[team].day_lag_snapshots.get(i, this_sim.teams[team].day_lag_snapshots['pre-season']) for i in elo.SNAPSHOT_LOOKBACKS]
		out_data.append(this_row + [round(this_row[-1] - snap) for snap in snaps])
	snap_col_names = [f'{i}-Day Change' for i in elo.SNAPSHOT_LOOKBACKS]
	out_data = sorted(out_data, key = lambda x: x[1], reverse = True)
	return pd.DataFrame(out_data, columns = ['Team', 'Elo Rating'] + snap_col_names)

def make_predictions(this_sim, df, pred_date = None):
	'''
	Defaults to making predictions for every game from today onwards. 
	pred_date could be used to specify a specific date to predict in format "YYYY-MM-DD"
	Output is a list of every game predicted in the format: 
	[Date, Away, Home, Away WinP, Home WinP, Away ML, Away Threshold, Home ML, Home Threshold]
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
					row['Away_ML'], odds_needed(1 - winph, ADV_TO_USE), row['Home_ML'], odds_needed(winph, ADV_TO_USE)])
	
	output_df = pd.DataFrame(preds, columns = ["Date", "Away", "Home", "Away WinP", "Home WinP", "Away ML", "Away Threshold", "Home ML", "Home Threshold"])
	utils.table_output(output_df, 'Game Predictions Based on Ratings through ' + this_sim.date)
	t = ADV_PCT_THRESHOLD if ADV_TO_USE == 'ADV_PCT' else ADV_THRESHOLD
	last_7_30_365 = (eval_recent_performance(7, ADV_TO_USE, t), eval_recent_performance(30, ADV_TO_USE, t), eval_recent_performance(365, ADV_TO_USE, t))
	
	#save the predictions output and ratings in markdown where github pages can find it
	ratings = clean_up_ratings(this_sim)
	utils.save_markdown_df(output_df, ratings, pred_date, last_7_30_365)

def main():
	'''
	This method leverages elo.py to get the latest old data, scrape to update, and get the odds for upcoming games
	Using that updated dataframe, this script predicts games today by default
	'''
	this_sim, df = elo.main(scrape = True, save_scrape = True, save_new_scrape = False, print_ratings = False)
	make_predictions(this_sim, df, pred_date = utils.date_to_string(datetime.today()))
	utils.clean_up_old_outputs_and_data()

if __name__ == '__main__':
	main()