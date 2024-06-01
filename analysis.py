import numpy as np
import pandas as pd
from predictions import ADV_THRESHOLD, ADV_PCT_THRESHOLD
from elo import sim, K_FACTOR, HOME_ADVANTAGE
from utils import get_latest_data_filepath, today_date_string, shift_dstring
UNIT = 1
cols = 'ADV_PCT' #toggle between ADV_PCT and ADV

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

def convert_to_betting_rows(combo):
    #convert dataframe from one row per game to one row per possible betting side (2 rows per game)
    mdf = pd.melt(combo, id_vars = ['H_WAGER', 'A_WAGER', 'H_PROFIT', 'A_PROFIT'], value_vars = [f'H_{cols}', f'A_{cols}'])
    mdf['WAGER'] = mdf.apply(lambda x: x['H_WAGER'] if x['variable'] == f'H_{cols}' else x['A_WAGER'], axis = 1)
    mdf['PROFIT'] = mdf.apply(lambda x: x['H_PROFIT'] if x['variable'] == f'H_{cols}' else x['A_PROFIT'], axis = 1)
    mdf.rename(columns = {'value': 'ADVANTAGE'}, inplace = True)
    return mdf

def eval_recent_performance(recent_days):
    combo = assemble_results_and_predictions()
    combo = combo.loc[combo['Date'] >= shift_dstring(today_date_string(), -recent_days)]
    
    mdf = convert_to_betting_rows(combo)
    total_games = mdf.shape[0] / 2 #divide by two since mdf has one row for each side of the game
    
    #return the ROI and bet rate with current threshold used for predictions
    aggs = mdf.loc[mdf['ADVANTAGE'] >= ADV_PCT_THRESHOLD][['WAGER', 'PROFIT']].agg(['size', 'sum'])
    return (round(aggs.loc['sum', 'PROFIT'] / aggs.loc['sum', 'WAGER'] * 100, 2), round(aggs.loc['size', 'WAGER'] / total_games * 100))

def tuning():
    '''
    Once realizing 538s ELOs are no longer going to be updated, I needed to replicate their ELO system
    script checks that there still is some signal to the scheme when using my ELOs. The point is to find
    the recommended threshold cutoffs for ADV and ADV_PCT based on historical performance
    '''
    combo = assemble_results_and_predictions()
    mdf = convert_to_betting_rows(combo)

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


#when ADV is percentage-based
'''
Performance for 2023 season
choose threshold 0.07 for 3.5% ROI and 31% bet rate
    adv_threshold  games_bet  winnings   ROI  percent_games_bet
0            0.01     1483.0     27.46  1.70                 67
1            0.02     1311.0     27.10  1.92                 59
2            0.03     1175.0      9.67  0.77                 53
3            0.04     1047.0     21.29  1.91                 47
4            0.05      909.0      2.91  0.30                 41
5            0.06      797.0      4.47  0.54                 36
6            0.07      688.0     24.69  3.47                 31
7            0.08      592.0     16.56  2.72                 27
8            0.09      495.0     11.84  2.34                 22
9            0.10      424.0      8.83  2.04                 19
10           0.11      344.0      7.38  2.10                 15
11           0.12      280.0     11.93  4.19                 13
12           0.13      215.0     11.69  5.37                 10
13           0.14      162.0      7.90  4.81                  7
14           0.15      119.0     -1.25 -1.04                  5
15           0.16       62.0      6.03  9.65                  3
16           0.17       30.0     -0.66 -2.18                  1
'''

#when ADV is difference-based
'''
Performance for 2023 season
choose threshold 0.04 for ~2.2% ROI and ~23% bet rate
    adv_threshold  games_bet  winnings    ROI  percent_games_bet
0            0.01     1294.0      9.68   0.69                 58
1            0.02     1015.0      5.24   0.48                 46
2            0.03      734.0      4.77   0.61                 33
3            0.04      505.0     11.57   2.20                 23
4            0.05      296.0     -5.10  -1.67                 13
5            0.06      148.0     -4.38  -2.89                  7
6            0.07       16.0     -3.87 -23.55                  1
'''
