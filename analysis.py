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
    homes = np.arange(14, 18, 1)
    ks = np.arange(3.5, 5.6, 0.5)

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

# advantage_cutoff_tuning('ADV', '2024-09-03')
# tune_home_and_k()
# kelly_tuning(adv_to_use = 'ADV_PCT', wager_type = 'half-kelly')

#when ADV is percentage-based
'''
Performance for 2023 through August 2024 
choose threshold 0.11 for 10.5% ROI and 14% bet rate
    adv_threshold  games_bet  winnings    ROI  percent_games_bet
0            0.01     2650.0    -26.91  -0.94                 65
1            0.02     2373.0    -35.30  -1.39                 58
2            0.03     2089.0    -36.25  -1.64                 51
3            0.04     1815.0      5.93   0.31                 44
4            0.05     1587.0     14.15   0.86                 39
5            0.06     1381.0      0.84   0.06                 34
6            0.07     1198.0     10.71   0.87                 29
7            0.08     1004.0     26.12   2.55                 25
8            0.09      836.0     22.94   2.69                 20
9            0.10      693.0     38.87   5.52                 17
10           0.11      562.0     59.50  10.46                 14
11           0.12      444.0     44.67   9.96                 11
12           0.13      341.0     24.13   7.02                  8
13           0.14      259.0     16.20   6.22                  6
14           0.15      182.0     -8.61  -4.70                  4
15           0.16       99.0      9.17   9.20                  2
16           0.17       38.0     12.13  31.63                  1
'''

#when ADV is difference-based
'''
Performance for 2023 through June 2024
choose threshold 0.05 for 5.4% ROI and 12% bet rate
    adv_threshold  games_bet  winnings    ROI  percent_games_bet
0            0.01     2348.0    -53.84  -2.14                 57
1            0.02     1759.0    -26.73  -1.44                 43
2            0.03     1248.0      1.89   0.15                 30
3            0.04      817.0     15.98   1.89                 20
4            0.05      474.0     26.42   5.42                 12
5            0.06      214.0     14.23   6.50                  5
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
