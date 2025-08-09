import os
import numpy as np
import pandas as pd
from elo import sim, K_FACTOR, HOME_ADVANTAGE
from utils import *
from tqdm import tqdm
from predictions import assemble_results_and_predictions, convert_to_betting_rows
import plotly.express as px

DIAG_DIR = 'OUTPUTS/diagnostics'

def advantage_cutoff_tuning(adv_to_use: str, end_early: str = None, start_late: str = None,
                            warmup_cutoff: str = '2019-07-01', save_outputs: bool = True) -> pd.DataFrame:
    '''
    Evaluate flat-bet ROI across a range of advantage thresholds for ADV or ADV_PCT.
    - Only games on/after warmup_cutoff are included in ROI metrics (earlier games seed Elo).
    - Returns a DataFrame of threshold, games bet, winnings, ROI, and bet rate.
    '''
    os.makedirs(DIAG_DIR, exist_ok=True)
    combo = assemble_results_and_predictions()
    # Ensure we only evaluate rows with resolved games (scores present)
    # keep only resolved games (numeric scores)
    combo = combo.loc[pd.to_numeric(combo['Home_Score'], errors='coerce').notna() &
                      pd.to_numeric(combo['Away_Score'], errors='coerce').notna()]
    # apply date filters
    if end_early is not None:
        combo = combo.loc[combo['Date'] <= end_early]
    if start_late is not None:
        combo = combo.loc[combo['Date'] >= start_late]
    if warmup_cutoff is not None:
        combo = combo.loc[combo['Date'] >= warmup_cutoff]

    mdf = convert_to_betting_rows(combo, adv_to_use)

    # robust trimming of extreme advantage tails; fall back gracefully on small samples
    try:
        mdf['ventile_rank'] = pd.qcut(mdf['ADVANTAGE'], 20, labels=False, duplicates='drop')
        if mdf['ventile_rank'].notna().any():
            mdf = mdf.loc[(mdf['ventile_rank'] > 0) & (mdf['ventile_rank'] < mdf['ventile_rank'].max())]
    except Exception:
        pass
    
    total_games = max(mdf.shape[0] / 2, 1)  # avoid divide-by-zero

    possible_triggers = np.round(np.arange(0.01, 0.201, 0.01), 2)
    rows = []
    for t in possible_triggers:
        sel = mdf.loc[mdf['ADVANTAGE'] >= t]
        if sel.empty or sel['WAGER'].sum() == 0:
            rows.append([t, 0, 0.0, 0.0, 0.0])
            continue
        aggs = sel[['WAGER', 'PROFIT']].agg(['size', 'sum'])
        roi = float(aggs.loc['sum', 'PROFIT'] / aggs.loc['sum', 'WAGER'] * 100)
        bet_rate = float(aggs.loc['size', 'WAGER'] / total_games * 100)
        rows.append([t, int(aggs.loc['size', 'WAGER']), float(aggs.loc['sum', 'PROFIT']), round(roi, 2), round(bet_rate, 2)])

    out = pd.DataFrame(rows, columns=['adv_threshold', 'games_bet', 'winnings', 'ROI', 'percent_games_bet'])
    print(out)

    # simple ROI curve
    if save_outputs:
        fig = px.line(out, x='adv_threshold', y='ROI', title=f'ROI vs Threshold ({adv_to_use})')
        fig_path = os.path.join(DIAG_DIR, f'roi_curve_{adv_to_use}.html')
        fig.write_html(fig_path)
    return out

'''
Performance for 7/30/23 through 8/6/25 using ADV_PCT 
choose threshold 0.11 for 3.7% ROI and 11% bet rate
    adv_threshold  games_bet  winnings   ROI  percent_games_bet
0            0.01       2686    -45.42 -1.55              60.70
1            0.02       2391    -53.06 -2.05              54.03
2            0.03       2078    -71.21 -3.19              46.96
3            0.04       1811    -73.69 -3.82              40.93
4            0.05       1557    -58.23 -3.55              35.19
5            0.06       1313    -34.12 -2.48              29.67
6            0.07       1104    -29.35 -2.57              24.95
7            0.08        920    -43.64 -4.60              20.79
8            0.09        774    -17.43 -2.19              17.49
9            0.10        622     18.92  2.97              14.06
10           0.11        487     18.32  3.70              11.01
11           0.12        365      4.42  1.19               8.25
12           0.13        271      1.42  0.52               6.12
13           0.14        187     10.60  5.61               4.23
14           0.15        112     -3.90 -3.47               2.53
15           0.16         52     -1.79 -3.43               1.18
'''

'''
Performance for 7/30/23 through 7/30/25 using ADV
choose threshold 0.06 for 2.8% ROI and 3.4% bet rate
    adv_threshold  games_bet  winnings   ROI  percent_games_bet
0            0.01       2365    -63.35 -2.46              53.45
1            0.02       1740    -81.25 -4.34              39.32
2            0.03       1187    -43.62 -3.46              26.82
3            0.04        763    -31.70 -3.98              17.24
4            0.05        437    -16.74 -3.68               9.88
5            0.06        150      4.33  2.78               3.39
'''

def kelly_tuning(adv_to_use: str = 'ADV_PCT', wager_type: str = 'kelly'):
    #The Kelly Criterion is a wagering methodology to maximize the long-term expected geometric growth rate of a bank-roll
    #'kelly': wager according to kelly criterion recommendation (ROLL * (win_p - (loss_p / profit_multiple)))
    #'half-kelly': wager 50% of the kelly criterion recommendation (ROLL * 0.5 * (win_p - (loss_p / profit_multiple)))
    #The goal of this function is to observe the historical performance of this betting strategy when combined with a betting advantage selection strategy
    #
    #FINDINGS: we don't see any combinations of Kelly/Half-Kelley and any threshold that result in positive ROI like our flat betting scheme does (above)
    k_map = {'kelly': 1.0, 'half-kelly': 0.5}
    combo = assemble_results_and_predictions()
    combo['H_P_MULT'] = combo.apply(lambda x: - 100 / x['Home_ML'] if x['Home_ML'] < 0 else x['Home_ML'] / 100, axis = 1)
    combo['A_P_MULT'] = combo.apply(lambda x: - 100 / x['Away_ML'] if x['Away_ML'] < 0 else x['Away_ML'] / 100, axis = 1)

    adv_profits = []
    for adv_t in np.arange(.01, .21, .01):
        triggered_games = combo.loc[(combo[f'H_{adv_to_use}'] >= adv_t) | (combo[f'A_{adv_to_use}'] >= adv_t)]
        wagered, profited = [], []
        roll = 1000
        for index, row in triggered_games.iterrows():
            # Kelly fraction f* = p - (1 - p)/b, where b is profit multiple
            p_away = float(row['AWAY_PRE_PROB'])
            p_home = float(row['HOME_PRE_PROB'])
            b_away = float(row['A_P_MULT'])
            b_home = float(row['H_P_MULT'])
            ka = (p_away - (1 - p_away) / b_away) * k_map[wager_type]
            kh = (p_home - (1 - p_home) / b_home) * k_map[wager_type]
            ka = max(0.0, ka)
            kh = max(0.0, kh)
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

if __name__ == '__main__':
    advantage_cutoff_tuning('ADV_PCT', start_late='2023-07-30', end_early='2025-08-06')  # last 2.5 seasons
    pass
