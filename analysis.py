import numpy as np
import pandas as pd
from elo import sim, K_FACTOR, HOME_ADVANTAGE
from utils import *
from tqdm import tqdm
from predictions import assemble_results_and_predictions, convert_to_betting_rows

def advantage_cutoff_tuning(adv_to_use, end_early = None, start_late = None):
    '''
    Once realizing 538s ELOs are no longer going to be updated, I needed to replicate their ELO system
    script checks that there still is some signal to the scheme when using my ELOs. The point is to find
    the recommended threshold cutoffs for ADV and ADV_PCT based on historical performance
    '''
    combo = assemble_results_and_predictions()
    if end_early != None:
        combo = combo.loc[combo['Date'] <= end_early]
    if start_late != None:
        combo = combo.loc[combo['Date'] >= start_late]
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
    '''
    latest_df = pd.read_csv(get_latest_data_filepath())
    homes = np.arange(18, 19, 1)
    ks = np.arange(4, 4.5, 0.5)

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

# tune_home_and_k()
#games through 7/30/25, brier = .241769
# advantage_cutoff_tuning('ADV', start_late = '2023-07-30') #last 2.5 seasons

'''
Performance for 7/30/23 through 7/30/25 using ADV_PCT 
choose threshold 0.11 for 3.4% ROI and 12% bet rate
    adv_threshold  games_bet  winnings    ROI  percent_games_bet
0            0.01     2659.0    -58.57  -2.04                 61
1            0.02     2357.0    -70.76  -2.81                 54
2            0.03     2068.0    -60.58  -2.76                 48
3            0.04     1812.0    -70.49  -3.71                 42
4            0.05     1558.0    -56.28  -3.47                 36
5            0.06     1327.0    -51.90  -3.79                 31
6            0.07     1128.0    -53.28  -4.60                 26
7            0.08      959.0    -25.27  -2.57                 22
8            0.09      801.0    -13.36  -1.64                 18
9            0.10      661.0      6.45   0.96                 15
10           0.11      516.0     17.72   3.40                 12
11           0.12      405.0    -10.62  -2.59                  9
12           0.13      300.0      2.04   0.68                  7
13           0.14      220.0     -6.39  -2.89                  5
14           0.15      142.0    -16.37 -11.49                  3
15           0.16       71.0     -6.57  -9.23                  2
16           0.17       20.0     -0.21  -1.05                  0
'''

#when ADV is difference-based
'''
Performance for 7/30/23 through 7/30/25
choose threshold 0.06 for 2.5% ROI and 4% bet rate
    adv_threshold  games_bet  winnings   ROI  percent_games_bet
0            0.01     2343.0    -81.64 -3.24                 54
1            0.02     1740.0    -92.45 -5.00                 40
2            0.03     1187.0    -40.92 -3.29                 27
3            0.04      766.0    -32.97 -4.16                 18
4            0.05      448.0     -1.16 -0.25                 10
5            0.06      174.0      4.42  2.49                  4
'''

def kelly_tuning(adv_to_use = 'ADV_PCT', wager_type = 'kelly'):
    #The Kelly Criterion is a wagering methodology to maximize the long-term expected geometric growth rate of a bank-roll
    #'kelly': wager according to kelly criterion recommendation (ROLL * (win_p - (loss_p / profit_multiple)))
    #'half-kelly': wager 50% of the kelly criterion recommendation (ROLL * 0.5 * (win_p - (loss_p / profit_multiple)))
    #The goal of this function is to observe the historical performance of this betting strategy when combined with a betting advantage selection strategy
    #
    #FINDINGS: we don't see any combinations of Kelly/Half-Kelley and any threshold that result in positive ROI like our flat betting scheme does (above)
    k_map = {'kelly': 1, 'half-kelly': 0.5}
    combo = assemble_results_and_predictions()
    combo['H_P_MULT'] = combo.apply(lambda x: - 100 / x['Home_ML'] if x['Home_ML'] < 0 else x['Home_ML'] / 100, axis = 1)
    combo['A_P_MULT'] = combo.apply(lambda x: - 100 / x['Away_ML'] if x['Away_ML'] < 0 else x['Away_ML'] / 100, axis = 1)

    adv_profits = []
    for adv_t in np.arange(.01, .21, .01):
        triggered_games = combo.loc[(combo[f'H_{adv_to_use}'] >= adv_t) | (combo[f'A_{adv_to_use}'] >= adv_t)]
        wagered, profited = [], []
        roll = 1000
        for index, row in triggered_games.iterrows():
            ka = (row['AWAY_PRE_PROB'] - row['HOME_PRE_PROB'] / row['A_P_MULT']) * k_map[wager_type]
            kh = (row['HOME_PRE_PROB'] - row['AWAY_PRE_PROB'] / row['H_P_MULT']) * k_map[wager_type]
            if row[f'A_{adv_to_use}'] >= adv_t:
                #proceed with betting on Away
                w = ka * roll
                p = w * row['A_P_MULT'] if int(row['Away_Score']) > int(row['Home_Score']) else - w
            else:
                #proceed with betting on Home
                w = kh * roll
                p = w * row['H_P_MULT'] if int(row['Away_Score']) < int(row['Home_Score']) else - w
            roll += p
            wagered.append(w)
            profited.append(p)

        triggered_games['K_WAGERED'] = wagered
        triggered_games['K_PROFITED'] = profited
        aggs = triggered_games[triggered_games['K_WAGERED'] > 0][['K_WAGERED', 'K_PROFITED']].agg(['size', 'sum'])
        adv_profits.append([adv_t, aggs.loc['size', 'K_WAGERED'], aggs.loc['sum', 'K_PROFITED'], round(aggs.loc['sum', 'K_PROFITED'] / aggs.loc['sum', 'K_WAGERED'] * 100, 2), round(aggs.loc['size', 'K_WAGERED'] / combo.shape[0] * 100)])
    print(pd.DataFrame(adv_profits, columns = ['trigger_threshold', 'games_bet', 'winnings', 'ROI', 'percent_games_bet']))    
