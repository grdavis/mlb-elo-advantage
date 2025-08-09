import os
import elo
import utils
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

WARMUP_CUTOFF_DATE = '2019-07-01'

class TuningSimulation(elo.ELO_Sim):
    def __init__(self):
        super().__init__()
        self.prediction_results = []
        self.betting_results = []
        self.season_count = 0
        
    def season_reset(self, season_carry):
        """Reset ratings between seasons"""
        for team in self.teams.values():
            team.elo = elo.ELO_BASE + (team.elo - elo.ELO_BASE) * season_carry
            team.season_wins = 0
            team.season_losses = 0
    
    def track_prediction(self, w_prob):
        """Track accuracy of win/loss predictions"""
        self.prediction_results.append(w_prob)

def evaluate_parameters(data, params, return_series: bool = False):
    """
    Evaluate a set of parameters by simulating historical games
    Returns various performance metrics. If return_series is True, also returns
    arrays of predicted home win probabilities and actual home outcomes.
    """
    this_sim = TuningSimulation()
    predicted_home_probs = []
    actual_home_results = []

    # Find last date with scores
    last_date = data[data['Home_Score'].notna()]['Date'].max()
    last_date = utils.string_to_date(last_date).date()

    for _, row in data.iterrows():
        bdate = row['Date']
        if utils.string_to_date(bdate).date() > last_date: break
        
        #apply the season reset and snapshot if we have incremented by a year
        if (this_sim.date != '') and (this_sim.date[:4] != bdate[:4]): 
            this_sim.season_reset(params['season_carry'])
        
        #step the Elo system forward based on the results in the row and track metrics
        this_sim.date = bdate
        home, away = row['Home'], row['Away']

        winner = home if int(row['Home_Score']) > int(row['Away_Score']) else away
        loser = home if int(row['Away_Score']) > int(row['Home_Score']) else away
        winnerScore = int(row['Home_Score']) if int(row['Home_Score']) > int(row['Away_Score']) else int(row['Away_Score'])
        loserScore = int(row['Away_Score']) if int(row['Home_Score']) > int(row['Away_Score']) else int(row['Home_Score'])

        #instantiate teams with base ELO of 1500 if we're on a team's first game in the system
        if home not in this_sim.teams: 
            this_sim.teams[home] = elo.Team(home, this_sim.date, elo.ELO_BASE)
        if away not in this_sim.teams: 
            this_sim.teams[away] = elo.Team(away, this_sim.date, elo.ELO_BASE)

        pre_home = this_sim.get_elo(home)
        pre_away = this_sim.get_elo(away)

        is_playoffs = row['Date'][5:7] in ['10', '11']
        k_factor = params['k_factor'] if not is_playoffs else params['k_factor'] + elo.PLAYOFF_K_EXTRA
        margin_mult = 1 if not is_playoffs else elo.PLAYOFF_MARGIN_MULT

        # compute predicted home win probability using parameterized home advantage
        elo_margin_pre = (pre_home + params['home_advantage'] - pre_away) * margin_mult
        home_pre_winp = 1 / (1 + 10 ** (-elo_margin_pre / 400))
        # only record metrics after warm-up cutoff; still always update ratings
        if bdate >= WARMUP_CUTOFF_DATE:
            predicted_home_probs.append(home_pre_winp)
            actual_home_results.append(1 if winner == home else 0)

        # update ratings based on realized winner (MoV-aware Elo)
        # add home advantage to the realized winner's side
        Welo_0, Lelo_0 = this_sim.get_elo(winner), this_sim.get_elo(loser)
        if winner == home:
            Welo_0 += params['home_advantage']
        else:
            Lelo_0 += params['home_advantage']
        elo_margin = (Welo_0 - Lelo_0) * margin_mult
        w_winp = 1 / (1 + 10 ** (-elo_margin / 400))

        MoV = winnerScore - loserScore #how much did winner win by
        MoV_multiplier = elo.calc_MoV_multiplier(elo_margin, MoV)
        elo_delta = k_factor * MoV_multiplier * (1 - w_winp)
        this_sim.update_elos(winner, loser, elo_delta)

    # Calculate performance metrics
    if len(predicted_home_probs) == 0:
        pred_accuracy = float('nan')
        brier_score = float('nan')
    else:
        # Classification accuracy at 50% threshold against actual home results
        pred_accuracy = float(np.mean(((np.array(predicted_home_probs) >= 0.5).astype(int) == np.array(actual_home_results)).astype(int)))
        # Proper Brier score for home win probability
        brier_score = float(np.mean((np.array(predicted_home_probs) - np.array(actual_home_results)) ** 2))

    metrics = {
        'prediction_accuracy': pred_accuracy,
        'brier_score': brier_score
    }
    if return_series:
        return metrics, np.array(predicted_home_probs), np.array(actual_home_results)
    return metrics

def ensure_output_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def grid_search():
    """
    Perform grid search over parameter combinations
    """
    # Load historical data
    data = pd.read_csv(utils.get_latest_data_filepath())
    # Ensure Date is string, then filter rows to before today (scores should exist)
    data['Date'] = data['Date'].astype(str)
    data = data.sort_values('Date')
    # Fast-mode: restrict to recent window to speed up runs
    fast_mode = os.environ.get('TUNING_FAST', '0') == '1'
    if fast_mode:
        days = int(os.environ.get('TUNING_DAYS', '30'))
        start_date = utils.shift_dstring(utils.today_date_string(), -days)
        data = data.loc[(data['Date'] >= start_date) & (data['Date'] < utils.today_date_string())]
        print(f"Fast mode ON: limiting data to last {days} days ({start_date}..{utils.today_date_string()}). Rows: {len(data)}")
    
    # Define parameter grid
    if fast_mode:
        param_grid = {
            'k_factor': [3.0, 3.5, 4.0, 4.5],
            'home_advantage': [14, 16, 18],
            'season_carry': [0.63, 0.65, 0.67],
        }
    else:
        param_grid = {
            'k_factor': [4.25, 4.5, 4.75],
            'home_advantage': [19, 20, 21, 22],
            'season_carry': [0.70, 0.71, 0.72, 0.73],
        }
    
    results = []
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Testing {total_combinations} parameter combinations...")
    
    # Generate all parameter combinations
    keys = param_grid.keys()
    for values in tqdm(product(*param_grid.values()), total=total_combinations):
        params = dict(zip(keys, values))
        
        # Evaluate this parameter set
        metrics = evaluate_parameters(data, params)
        
        # Store results
        results.append({
            **params,
            **metrics
        })
        
    results_df = pd.DataFrame(results)
    
    # Sort by different metrics
    print("\nTop 20 by prediction accuracy:")
    print(results_df.sort_values('prediction_accuracy', ascending=False).head(20))
    
    print("\nTop 20 by brier score:")
    print(results_df.sort_values('brier_score', ascending=True).head(20))
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'tuning_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Saved raw results to {results_path}")

    # Visualizations
    out_dir = os.path.join('OUTPUTS', 'tuning')
    ensure_output_dir(out_dir)

    # 1) Heatmap: best brier per (k_factor, home_advantage)
    best_brier = (results_df
                  .groupby(['k_factor', 'home_advantage'], as_index=False)
                  .agg(brier_score=('brier_score', 'min')))
    heat_pivot = best_brier.pivot(index='k_factor', columns='home_advantage', values='brier_score')
    heat_fig = px.imshow(
        heat_pivot.values,
        x=heat_pivot.columns.astype(str),
        y=heat_pivot.index.astype(str),
        color_continuous_scale='Viridis',
        labels=dict(x='home_advantage', y='k_factor', color='brier_score'),
        title='Brier Score Heatmap (min over season_carry)'
    )
    heat_fig.update_layout(yaxis_autorange='reversed')
    heat_path = os.path.join(out_dir, f'brier_heatmap_{timestamp}.html')
    heat_fig.write_html(heat_path)
    print(f"Wrote {heat_path}")

    # 2) Season carry vs brier (mean across k,home)
    sc_series = results_df.groupby('season_carry', as_index=False)['brier_score'].mean()
    sc_fig = px.line(sc_series, x='season_carry', y='brier_score', title='Average Brier vs Season Carry')
    sc_path = os.path.join(out_dir, f'season_carry_curve_{timestamp}.html')
    sc_fig.write_html(sc_path)
    print(f"Wrote {sc_path}")

    # 3) Select best params by lowest Brier and make a calibration plot
    best_row = results_df.sort_values('brier_score', ascending=True).iloc[0].to_dict()
    best_params = {
        'k_factor': float(best_row['k_factor']),
        'home_advantage': float(best_row['home_advantage']),
        'season_carry': float(best_row['season_carry']),
    }
    print(f"Best params by Brier: {best_params}")

    metrics, preds, outcomes = evaluate_parameters(data, best_params, return_series=True)
    # Bin into 1% buckets
    bins = np.linspace(0.0, 1.0, 101)
    bucket = pd.cut(preds, bins=bins, include_lowest=True)
    cal_df = pd.DataFrame({'pred': preds, 'obs': outcomes, 'bucket': bucket})
    cal_g = cal_df.groupby('bucket').agg(mean_pred=('pred', 'mean'), emp_rate=('obs', 'mean'), size=('obs', 'size')).reset_index()
    cal_fig = go.Figure()
    cal_fig.add_trace(go.Scatter(x=cal_g['mean_pred'], y=cal_g['emp_rate'], mode='markers', name='Empirical'))
    cal_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Ideal', line=dict(dash='dash')))
    cal_fig.update_layout(title=f'Calibration (Best Params) — Brier={metrics["brier_score"]:.4f}',
                          xaxis_title='Predicted Home Win Prob', yaxis_title='Empirical Home Win Rate')
    cal_path = os.path.join(out_dir, f'calibration_{timestamp}.html')
    cal_fig.write_html(cal_path)
    print(f"Wrote {cal_path}")

    # Print a suggested override for constants based on best params (for manual review)
    print("\nSuggested constants (manual review before changing code):")
    print(f"K_FACTOR = {best_params['k_factor']}")
    print(f"HOME_ADVANTAGE = {int(round(best_params['home_advantage']))}")
    print(f"SEASON_RESET_MULT = {best_params['season_carry']}")

    return results_df

if __name__ == '__main__':
    # games through 7/30/25, brier = .241640
    grid_search() 