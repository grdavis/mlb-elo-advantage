import numpy as np
import pandas as pd
from elo import sim, K_FACTOR, HOME_ADVANTAGE
from utils import get_latest_data_filepath, today_date_string, shift_dstring
from tqdm import tqdm
import predictions

UNIT = 1

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

def eval_recent_performance(recent_days, adv_to_use):
    combo = assemble_results_and_predictions()
    combo = combo.loc[combo['Date'] >= shift_dstring(today_date_string(), -recent_days)]
    
    mdf = convert_to_betting_rows(combo, adv_to_use)
    total_games = mdf.shape[0] / 2 #divide by two since mdf has one row for each side of the game
    
    threshold = predictions.ADV_PCT_THRESHOLD if adv_to_use != 'adv' else predictions.ADV_THRESHOLD

    #return the ROI and bet rate with current threshold used for predictions
    aggs = mdf.loc[mdf['ADVANTAGE'] >= threshold][['WAGER', 'PROFIT']].agg(['size', 'sum'])
    return (round(aggs.loc['sum', 'PROFIT'] / aggs.loc['sum', 'WAGER'] * 100, 2), round(aggs.loc['size', 'WAGER'] / total_games * 100))

def advantage_cutoff_tuning(adv_to_use, end_early = None):
    '''
    Once realizing 538s ELOs are no longer going to be updated, I needed to replicate their ELO system
    script checks that there still is some signal to the scheme when using my ELOs. The point is to find
    the recommended threshold cutoffs for ADV and ADV_PCT based on historical performance
    '''
    combo = assemble_results_and_predictions()
    if end_early != None:
        combo = combo.loc[combo['Date'] <= end_early]
    mdf = convert_to_betting_rows(combo, adv_to_use)

    #there might be some outliers throwing off the numbers, so let's remove the top 5% and bottom 5% of thresholds
    mdf['decile_rank'] = pd.qcut(mdf['ADVANTAGE'], 20, labels = False)
    mdf = mdf.loc[(mdf['decile_rank'] > 0) & (mdf['decile_rank'] < 19)]
    
    total_games = mdf.shape[0] / 2 #divide by two since mdf has one row for each side of the game

    possible_triggers = np.arange(.01, .21, .01)
    adv_profits = []
    for t in possible_triggers:
        aggs = mdf.loc[mdf['ADVANTAGE'] >= t][['WAGER', 'PROFIT']].agg(['size', 'sum'])
        adv_profits.append([round(t, 2), aggs.loc['size', 'WAGER'], aggs.loc['sum', 'PROFIT'], round(aggs.loc['sum', 'PROFIT'] / aggs.loc['sum', 'WAGER'] * 100, 2), round(aggs.loc['size', 'WAGER'] / total_games * 100)])
    print(pd.DataFrame(adv_profits, columns = ['adv_threshold', 'games_bet', 'winnings', 'ROI', 'percent_games_bet']))

def tune_home_and_k():
    '''
    The original K Factor and Home Advantage were selected by copying the recommendations from 538 (4 and 24, respectively).
    Those parameters were likely selected by backtesting on a much larger set of data than what we have since 2023, but 
    we should still check if these parameters are still reasonable

    UPDATE 6/7/24: based on some findings using this function and a historical analysis of games since the start of the 2022
    season, we are updating home advantage to be 17. This implies the average home team wins 52.44% of games which closely
    aligns with the 52.437% home win percentage observed over these years and 6484 games. Since the pandemic, home field advantage
    in MLB has seemed to decline: https://www.washingtonpost.com/sports/2023/05/11/baseball-home-field-advantage/
    And it's not unique to baseball: https://sports.yahoo.com/farewell-to-nfl-home-field-advantage-as-home-teams-have-a-losing-record-through-5-weeks-153759757.html
    '''
    latest_df = pd.read_csv(get_latest_data_filepath())
    homes = np.arange(8, 20, 3)
    ks = np.arange(3.5, 5.5, 0.5)

    #Calculate and print out the brier score for each combination. We want brier to be LOW
    briers = []
    for h in tqdm(homes):
        for k in ks:
            this_sim, combo = sim(latest_df, k, h)
            combo = combo.loc[~combo['HOME_PRE_PROB'].isna()]
            combo['home_win'] = combo.apply(lambda x: 1 if x['Home_Score'] > x['Away_Score'] else 0, axis = 1)
            combo['brier'] = (combo['HOME_PRE_PROB'] - combo['home_win'])**2
            briers.append((h, k, combo['brier'].mean()))

    print(sorted(briers, key = lambda x: x[-1]))

# advantage_cutoff_tuning('adv', '2024-01-01')
# tune_home_and_k()
#when ADV is percentage-based
'''
Performance for 2023 through June 2024 
choose threshold 0.08 for 5.2% ROI and 26% bet rate
    adv_threshold  games_bet  winnings    ROI  percent_games_bet
0            0.01     2042.0    -12.52  -0.57                 67
1            0.02     1830.0     11.04   0.57                 60
2            0.03     1612.0      0.20   0.01                 53
3            0.04     1410.0     17.01   1.15                 46
4            0.05     1236.0     20.39   1.59                 40
5            0.06     1081.0     21.09   1.89                 35
6            0.07      934.0     23.64   2.47                 30
7            0.08      805.0     42.97   5.24                 26
8            0.09      671.0     29.86   4.37                 22
9            0.10      570.0     47.12   8.14                 19
10           0.11      458.0     37.64   8.12                 15
11           0.12      371.0     33.85   9.06                 12
12           0.13      293.0     33.58  11.39                 10
13           0.14      221.0      2.09   0.94                  7
14           0.15      158.0     -2.76  -1.74                  5
15           0.16       84.0     -9.15 -10.85                  3
16           0.17       43.0     -9.25 -21.51                  1
'''

#when ADV is difference-based
'''
Performance for 2023 through June 2024
choose threshold 0.04 for 7.6% ROI and 22% bet rate
    adv_threshold  games_bet  winnings    ROI  percent_games_bet
0            0.01     1349.0     21.58   1.49                 61
1            0.02     1015.0     17.27   1.60                 46
2            0.03      750.0     21.11   2.70                 34
3            0.04      499.0     39.00   7.58                 22
4            0.05      306.0     35.11  11.22                 14
5            0.06      154.0      2.69   1.72                  7
6            0.07       32.0     -5.03 -15.62                  1
'''
