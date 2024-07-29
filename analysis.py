import numpy as np
import pandas as pd
from elo import sim, K_FACTOR, HOME_ADVANTAGE
from utils import *
from tqdm import tqdm

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

# advantage_cutoff_tuning('ADV_PCT', '2024-07-26')
# tune_home_and_k()

#when ADV is percentage-based
'''
Performance for 2023 through June 2024 
choose threshold 0.11 for 10.6% ROI and 14% bet rate
    adv_threshold  games_bet  winnings    ROI  percent_games_bet
0            0.01     2365.0     -6.87  -0.27                 65
1            0.02     2124.0    -10.61  -0.47                 59
2            0.03     1873.0    -19.47  -0.98                 52
3            0.04     1636.0     14.89   0.87                 45
4            0.05     1427.0     14.72   0.99                 39
5            0.06     1241.0      2.69   0.21                 34
6            0.07     1077.0     13.00   1.18                 30
7            0.08      906.0     31.71   3.43                 25
8            0.09      762.0     27.97   3.60                 21
9            0.10      632.0     36.98   5.76                 17
10           0.11      516.0     55.47  10.63                 14
11           0.12      414.0     41.82  10.02                 11
12           0.13      320.0     24.64   7.65                  9
13           0.14      245.0     18.93   7.69                  7
14           0.15      171.0     -6.88  -4.00                  5
15           0.16       96.0     10.14  10.49                  3
16           0.17       39.0      9.10  23.13                  1
'''

#when ADV is difference-based
'''
Performance for 2023 through June 2024
choose threshold 0.04 for 3.1% ROI and 21% bet rate
    adv_threshold  games_bet  winnings    ROI  percent_games_bet
0            0.01     2100.0    -24.67  -1.10                 58
1            0.02     1579.0    -10.77  -0.64                 44
2            0.03     1125.0      9.46   0.81                 31
3            0.04      745.0     23.79   3.09                 21
4            0.05      433.0     24.66   5.54                 12
5            0.06      201.0     12.97   6.32                  6
6            0.07       25.0     -3.64 -14.17                  1
'''
