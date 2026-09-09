import os
import numpy as np
import pandas as pd
from elo import sim, K_FACTOR, HOME_ADVANTAGE
from utils import *
from tqdm import tqdm
from predictions import (
    assemble_results_and_predictions, convert_to_betting_rows, qualifying_bets,
    ADV_TO_USE, ADV_THRESHOLD, MAX_UNDERDOG_ML,
)
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
        sel = qualifying_bets(mdf, t) if adv_to_use == 'ADV' else mdf.loc[mdf['ADVANTAGE'] >= t]
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
Diagnosis (2026-09): rolling 365-day ROI of ADV_PCT>=0.11 collapsed to about -7% (30-day ~-24%).
This was not a one-month fluke.

What went wrong
- Team Elo is strictly worse than the moneyline market on Brier and accuracy every season.
  Market residual vs Elo is largely starting-pitcher quality (corr ~0.45 with Game Score gap).
- ADV_PCT inflates edges on longshots. Those bets were disproportionately away underdogs
  whose listed "value" was an ace vs a replacement starter the book already priced.
- The published 3.7% (7/30/23-8/6/25 at 0.11) was in-sample, and analysis.py trimmed the
  top/bottom 5% of advantages while production did not. Recomputed untrimmed ADV_PCT>=0.11
  on that same window is about -1.6%. 2023 was the only clearly positive year; 2024-2026
  reverted toward a slightly negative expectation as vig on ScoresAndOdds rose (~2% to ~4%).
- Aug 2025 K/HA retune (Brier on all history) did not create the collapse and is left as-is.

Fix (walk-forward; thresholds locked on 2019-2023 only)
- Add 538-style causal rolling Game Score adjustment to pre-game win probabilities.
- Bet absolute ADV >= 0.05, and never bet ML longer than +165.
- Keep the first stored moneyline snapshot instead of overwriting with later live cells.

Honest holdout (pitcher + ADV>=0.05 + ML<=+165): 2024 ~-1.5%, 2025 ~+1.7%, 2026 YTD ~+0.8%,
trailing 365 ~+0.7% vs previous 365 ~-7.4% on ADV_PCT 0.11 without pitchers. Train years remain
slightly negative; this is not a claim of a large market-beating edge.
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

def production_strategy_report():
    '''Print yearly ROI of the production rule (pitcher-adjusted ADV + longshot cap).'''
    combo = assemble_results_and_predictions()
    combo['year'] = combo['Date'].str[:4]
    print(f'Production rule: {ADV_TO_USE}>={ADV_THRESHOLD} and ML<={MAX_UNDERDOG_ML}')
    rows = []
    for label, mask in [
        ('2019-2023', combo['Date'].between('2019-01-01', '2023-12-31')),
        ('2024', combo['year'] == '2024'),
        ('2025', combo['year'] == '2025'),
        ('2026', combo['year'] == '2026'),
        ('trailing_365', combo['Date'] >= '2025-09-09'),
        ('trailing_30', combo['Date'] >= '2026-08-09'),
    ]:
        g = combo.loc[mask]
        if g.empty:
            continue
        mdf = convert_to_betting_rows(g, ADV_TO_USE)
        sel = qualifying_bets(mdf, ADV_THRESHOLD)
        w = sel['WAGER'].sum() if len(sel) else 0
        p = sel['PROFIT'].sum() if len(sel) else 0
        n = len(sel)
        roi = (p / w * 100) if w else float('nan')
        rows.append([label, len(g), n, round(n / max(len(g), 1) * 100, 1), round(p, 2), round(roi, 2)])
    print(pd.DataFrame(rows, columns=['window', 'games', 'bets', 'bet_rate', 'profit', 'ROI']))


if __name__ == '__main__':
    production_strategy_report()
