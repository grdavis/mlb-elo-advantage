import elo
import utils
from datetime import datetime
import pandas as pd
from season_predictions import get_playoff_probs
import pitcher_model

OUTPUT_PATH = f"OUTPUTS/{utils.date_to_string(datetime.today())[:10]} Game Predictions.csv"
# Absolute edge vs the bettable (vigged) implied probability. ADV_PCT was retired
# because it overweighted longshots; those were mostly starter mismatches the
# market already had in the price. Threshold locked on 2019-2023 only.
ADV_THRESHOLD = .05
ADV_PCT_THRESHOLD = .11
ADV_TO_USE = 'ADV'
# Do not bet sides longer than this. Break-even at +165 is ~37.7%; beyond that
# the pitcher-adjusted Elo tail is poorly calibrated against closing-ish lines.
MAX_UNDERDOG_ML = 165

def assemble_results_and_predictions():
    #grab the ratings and predictions for every historical game then enrich the dataset
    latest_df = pd.read_csv(utils.get_latest_data_filepath())
    latest_sim, combo = elo.sim(latest_df, elo.K_FACTOR, elo.HOME_ADVANTAGE)
    combo = combo.loc[combo['Date'] < utils.today_date_string()]
    combo = combo.loc[pd.to_numeric(combo['HOME_PRE_PROB'], errors='coerce').notna()]
    combo = combo.loc[pd.to_numeric(combo['Home_Score'], errors='coerce').notna() &
                      pd.to_numeric(combo['Away_Score'], errors='coerce').notna() &
                      pd.to_numeric(combo['Home_ML'], errors='coerce').notna() &
                      pd.to_numeric(combo['Away_ML'], errors='coerce').notna()]

    # Starter Game Score adjustment (538-style). Team Elo does not know who is
    # pitching; the betting market does. Without this, "value" bets clustered on
    # weak-starter underdogs and lost to the book.
    # Historical ROI uses the same on-disk pitcher cache the daily refresh writes.
    starters, rolling = pitcher_model.ensure_pitcher_cache()
    combo = pitcher_model.attach_pitchers_to_games(combo, starters, rolling)
    combo = pitcher_model.apply_pitcher_probs(combo, scale=pitcher_model.DEFAULT_GS_TO_ELO)

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
    combo['H_PROFIT'] = combo.apply(lambda x: utils.profit_calc(x['Home_ML'], x['H_WAGER']) if int(x['Home_Score']) > int(x['Away_Score']) else -x['H_WAGER'], axis = 1)
    combo['A_PROFIT'] = combo.apply(lambda x: utils.profit_calc(x['Away_ML'], x['A_WAGER']) if int(x['Away_Score']) > int(x['Home_Score']) else -x['A_WAGER'], axis = 1)

    return combo

def convert_to_betting_rows(combo, adv_to_use):
    #convert dataframe from one row per game to one row per possible betting side (2 rows per game)
    mdf = pd.melt(
        combo,
        id_vars=['H_WAGER', 'A_WAGER', 'H_PROFIT', 'A_PROFIT', 'Home_ML', 'Away_ML'],
        value_vars=[f'H_{adv_to_use}', f'A_{adv_to_use}'],
    )
    is_home = mdf['variable'] == f'H_{adv_to_use}'
    mdf['WAGER'] = mdf['H_WAGER'].where(is_home, mdf['A_WAGER'])
    mdf['PROFIT'] = mdf['H_PROFIT'].where(is_home, mdf['A_PROFIT'])
    mdf['ML'] = mdf['Home_ML'].where(is_home, mdf['Away_ML'])
    mdf.rename(columns={'value': 'ADVANTAGE'}, inplace=True)
    return mdf

def qualifying_bets(mdf, threshold):
    '''Advantage threshold plus the longshot cap used in production.'''
    return mdf.loc[(mdf['ADVANTAGE'] >= threshold) & (mdf['ML'] <= MAX_UNDERDOG_ML)]

def eval_recent_performance(recent_days, adv_to_use, threshold, combo=None):
    #evaluate performance of betting thresholds over recent days
    if combo is None:
        combo = assemble_results_and_predictions()
    combo = combo.loc[combo['Date'] >= utils.shift_dstring(utils.today_date_string(), -recent_days)]
    
	#return 0, 0 if there are no games to evaluate
    if combo.shape[0] == 0: return (0, 0)

    mdf = convert_to_betting_rows(combo, adv_to_use)
    total_games = mdf.shape[0] / 2 #divide by two since mdf has one row for each side of the game

    #return the ROI and bet rate with current threshold used for predictions
    sel = qualifying_bets(mdf, threshold)
    if sel.empty or sel['WAGER'].sum() == 0:
        return (0, 0)
    aggs = sel[['WAGER', 'PROFIT']].agg(['size', 'sum'])
    return (round(aggs.loc['sum', 'PROFIT'] / aggs.loc['sum', 'WAGER'] * 100, 2), round(aggs.loc['size', 'WAGER'] / total_games * 100))

def odds_needed(winp, adv_type):
	'''
	Given a pitcher-adjusted Elo win probability, return the moneyline we need
	to have ADV_THRESHOLD of absolute edge vs the bettable implied probability.
	Sides that would only clear the edge at prices longer than MAX_UNDERDOG_ML
	are marked n/a (we do not bet those longshots).
	'''
	if adv_type == 'ADV':
		original_implied = (winp - ADV_THRESHOLD)
	else:
		original_implied = (winp / (ADV_PCT_THRESHOLD + 1))

	if original_implied <= 0 or original_implied >= 1:
		return 'n/a'
	if original_implied <= .5:
		plus = int(round((100 / original_implied) - 100, 0))
		# No price we will actually bet meets the edge requirement
		if plus > MAX_UNDERDOG_ML:
			return 'n/a'
		return f"+{plus}"
	else:
		return f"-{int(round((100 * original_implied) / (1 - original_implied), 0))}"

def clean_up_ratings(this_sim, game_data):
	'''
	This function returns a DataFrame with one row for each team displaying their current Elo rating and how it has changed
	in the last 7 and 30. In addition, it shows the teams chances of making the playoffs, winning the division, reaching the
	divisional round, reaching the championship round, reaching the WS, and winning the WS. These probabilities are achieved
	by simulating all the remaining games and the playoffs many times (see season_predictions.MIN_PLAYOFF_SIMS)
	'''
	out_data = []
	for team in this_sim.teams:
		this_row = [team, round(this_sim.get_elo(team))]
		snaps = [this_sim.teams[team].day_lag_snapshots.get(i, this_sim.teams[team].day_lag_snapshots['pre-season']) for i in elo.SNAPSHOT_LOOKBACKS]
		out_data.append(this_row + [round(this_row[-1] - snap) for snap in snaps])
	snap_col_names = [f'{i}-Day Change' for i in elo.SNAPSHOT_LOOKBACKS]
	out_data = sorted(out_data, key = lambda x: x[1], reverse = True)
	ratings_df = pd.DataFrame(out_data, columns = ['Team', 'Elo Rating'] + snap_col_names)
	season_probs, n_sims = get_playoff_probs(this_sim, game_data)
	return pd.merge(ratings_df, season_probs, on = 'Team', how = 'inner'), n_sims

def make_predictions(this_sim, df, pred_date = None, starters=None, rolling=None):
	'''
	Defaults to making predictions for every game from today onwards. 
	pred_date could be used to specify a specific date to predict in format "YYYY-MM-DD"
	Output is a list of every game predicted in the format: 
	[Date, Away, Home, Away Pitcher, Home Pitcher, Away WinP, Home WinP, Away ML, Away Threshold, Home ML, Home Threshold]
	Output presented as a plotly table and saved to markdown for presentation on GitHub pages
	'''
	if starters is None or rolling is None:
		starters, rolling = pitcher_model.ensure_pitcher_cache()
	keyed = df.copy()
	keyed['matchup_on_date'] = keyed.groupby(['Date', 'Home', 'Away']).cumcount() + 1
	attached = pitcher_model.attach_pitchers_to_games(keyed, starters, rolling)
	adj_map = {}
	pitcher_names = {}
	for _, r in attached.iterrows():
		key = (str(r['Date'])[:10], r['Home'], r['Away'], int(r['matchup_on_date']))
		adj_map[key] = pitcher_model.pitcher_elo_adj(r['home_rolling_gs'], r['away_rolling_gs'])
		pitcher_names[key] = (r.get('away_pitcher') or '', r.get('home_pitcher') or '')

	preds = []
	for index, row in keyed.iterrows():
		if pred_date == None:
			#if no prediction date specified, skip over dates prior to today
			if utils.string_to_date(row['Date']).date() < datetime.today().date(): continue
		else:
			#if prediction date specified, skip over all dates not equal to the prediction date
			if row['Date'] != pred_date: continue
		is_playoffs = True if row['Date'][5:7] in ['10', '11'] else False
		key = (str(row['Date'])[:10], row['Home'], row['Away'], int(row['matchup_on_date']))
		adj = adj_map.get(key, 0.0)
		away_p, home_p = pitcher_names.get(key, ('', ''))
		if pd.isna(away_p):
			away_p = ''
		if pd.isna(home_p):
			home_p = ''
		winph = this_sim.predict_home_winp(row['Home'], row['Away'], is_playoffs, pitcher_adj=adj)
		preds.append([row['Date'], row['Away'], row['Home'], away_p, home_p,
					round(100-winph*100, 2), round(100*winph, 2), 
					row['Away_ML'], odds_needed(1 - winph, ADV_TO_USE), row['Home_ML'], odds_needed(winph, ADV_TO_USE)])
	
	output_df = pd.DataFrame(preds, columns = ["Date", "Away", "Home", "Away Pitcher", "Home Pitcher", "Away WinP", "Home WinP", "Away ML", "Away Threshold", "Home ML", "Home Threshold"])
	utils.table_output(output_df, 'Game Predictions Based on Ratings through ' + this_sim.date)
	t = ADV_PCT_THRESHOLD if ADV_TO_USE == 'ADV_PCT' else ADV_THRESHOLD
	combo = assemble_results_and_predictions()
	last_7_30_365 = (
		eval_recent_performance(7, ADV_TO_USE, t, combo=combo),
		eval_recent_performance(30, ADV_TO_USE, t, combo=combo),
		eval_recent_performance(365, ADV_TO_USE, t, combo=combo),
	)
	
	#save the predictions output and ratings in markdown where github pages can find it
	ratings, n_sims = clean_up_ratings(this_sim, df)
	utils.save_markdown_df(output_df, ratings, pred_date, last_7_30_365, sims = n_sims)

def main():
	'''
	This method leverages elo.py to get the latest old data, scrape to update, and get the odds for upcoming games
	Using that updated dataframe, this script predicts games today by default
	'''
	this_sim, df = elo.main(scrape = True, save_scrape = True, save_new_scrape = False, print_ratings = False)
	try:
		starters, rolling = pitcher_model.refresh_pitcher_cache()
	except Exception as e:
		print(f'Pitcher cache refresh failed ({e}); using existing cache')
		starters, rolling = pitcher_model.ensure_pitcher_cache()
	make_predictions(this_sim, df, pred_date = utils.date_to_string(datetime.today()),
					 starters=starters, rolling=rolling)
	utils.clean_up_old_outputs_and_data()

if __name__ == '__main__':
	main()