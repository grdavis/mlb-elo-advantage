import numpy as np
import pandas as pd
from elo import sim, K_FACTOR, HOME_ADVANTAGE
from utils import *
from tqdm import tqdm
from predictions import assemble_results_and_predictions, convert_to_betting_rows

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

    UPDATE 7/28/24: based on some findings using this function and a historical analysis of games since the start of the 2023
    season, we are updating home advantage to be 15. This implies the average home team wins 52.16% of games which closely
    aligns with the 52.19% home win percentage observed over these years and 4,045 games. Since the pandemic, home field advantage
    in MLB has seemed to decline: https://www.washingtonpost.com/sports/2023/05/11/baseball-home-field-advantage/
    And it's not unique to baseball: https://sports.yahoo.com/farewell-to-nfl-home-field-advantage-as-home-teams-have-a-losing-record-through-5-weeks-153759757.html
    '''
    latest_df = pd.read_csv(get_latest_data_filepath())
    homes = np.arange(15, 16, 1)
    ks = np.arange(3, 4.5, 0.5)

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
#games through 9/23/24, brier = .243769
# advantage_cutoff_tuning('ADV_PCT')

'''
Performance for 2023 through end of 2024 season using ADV_PCT 
choose threshold 0.11 for 9.4% ROI and 14% bet rate
    adv_threshold  games_bet  winnings   ROI  percent_games_bet
0            0.01     2862.0    -45.03 -1.46                 64
1            0.02     2564.0    -48.30 -1.76                 58
2            0.03     2264.0    -59.00 -2.46                 51
3            0.04     1962.0    -22.26 -1.08                 44
4            0.05     1709.0      2.64  0.15                 38
5            0.06     1485.0    -14.34 -0.93                 33
6            0.07     1270.0     -4.41 -0.34                 29
7            0.08     1080.0     14.36  1.30                 24
8            0.09      906.0     29.56  3.21                 20
9            0.10      743.0     41.48  5.50                 17
10           0.11      610.0     58.04  9.42                 14
11           0.12      479.0     45.41  9.40                 11
12           0.13      374.0     15.07  4.01                  8
13           0.14      282.0     12.25  4.32                  6
14           0.15      199.0     -1.27 -0.64                  4
15           0.16      113.0      9.29  8.20                  3
16           0.17       42.0      1.31  3.12                  1
'''

#when ADV is difference-based
'''
Performance for 2023 through end of 2024 season
choose threshold 0.05 for 3.4% ROI and 12% bet rate
    adv_threshold  games_bet  winnings    ROI  percent_games_bet
0            0.01     2542.0    -64.60  -2.37                 57
1            0.02     1898.0    -47.79  -2.37                 43
2            0.03     1345.0    -12.24  -0.87                 30
3            0.04      886.0      8.17   0.89                 20
4            0.05      527.0     18.56   3.42                 12
5            0.06      232.0      8.25   3.49                  5
6            0.07       16.0     -5.40 -32.91                  0
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
