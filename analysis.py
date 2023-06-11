import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('cleaned_logs_2019_to_20230605.csv')
cols = 'ADV_PCT'
mdf = pd.melt(df, id_vars = ['H_PROFIT', 'A_PROFIT'], value_vars = [f'H_{cols}', f'A_{cols}'])
mdf['PROFIT'] = mdf.apply(lambda x: x['H_PROFIT'] if x['variable'] == f'H_{cols}' else x['A_PROFIT'], axis = 1)
mdf.rename(columns = {'value': 'ADV'}, inplace = True)

#there might be some outliers throwing off the numbers, so let's remove the top 5% and bottom 5% of thresholds
mdf['decile_rank'] = pd.qcut(mdf['ADV'], 20, labels = False)
mdf = mdf.loc[(mdf['decile_rank'] > 0) & (mdf['decile_rank'] < 19)]

#update total games to remove 10%
total_games = df.shape[0] * .9

possible_triggers = np.arange(.01, .21, .01)
adv_profits = []
for t in possible_triggers:
	aggs = mdf.loc[mdf['ADV'] >= t]['PROFIT'].agg(['size', 'sum'])
	adv_profits.append([round(t, 2), aggs['size'], aggs['sum'], round(aggs['sum'] / aggs['size'] * 100, 2), round(aggs['size'] / total_games * 100)])
print(pd.DataFrame(adv_profits, columns = ['adv_threshold', 'games_bet', 'winnings', 'ROI', 'percent_games_bet']))

#when ADV is percentage-based
'''
    adv_threshold  games_bet    winnings    ROI  percent_games_bet
0            0.01     6744.0  196.734669   2.92                 81
1            0.02     5877.0  157.528955   2.68                 71
2            0.03     5047.0  162.100390   3.21                 61
3            0.04     4263.0  132.249361   3.10                 51
4            0.05     3554.0  134.296611   3.78                 43
5            0.06     2944.0  106.517764   3.62                 35
6            0.07     2409.0   79.597267   3.30                 29
7            0.08     1944.0  122.536597   6.30                 23
8            0.09     1543.0  114.963330   7.45                 19
9            0.10     1190.0  117.049023   9.84                 14
10           0.11      868.0   87.879762  10.12                 10
11           0.12      584.0   70.519340  12.08                  7
12           0.13      342.0   54.232131  15.86                  4
13           0.14      140.0   23.887955  17.06                  2
'''

#when ADV is difference-based
'''
    adv_threshold  games_bet    winnings   ROI  percent_games_bet
0            0.01     5802.0  149.945012  2.58                 70
1            0.02     4028.0  108.480934  2.69                 48
2            0.03     2577.0   63.369452  2.46                 31
3            0.04     1406.0   73.239346  5.21                 17
4            0.05      609.0   30.217414  4.96                  7
'''