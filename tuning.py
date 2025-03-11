import elo
import utils
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

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

def evaluate_parameters(data, params):
    """
    Evaluate a set of parameters by simulating historical games
    Returns various performance metrics
    """
    this_sim = TuningSimulation()
    output = []
    today_date = datetime.today().date()

    for _, row in data.iterrows():
        bdate = row['Date']
        if utils.string_to_date(bdate).date() >= today_date: break
        
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

        #instantiate teams with hard-coded Elos if we're on a team's first game in the system
        if home not in this_sim.teams: 
            this_sim.teams[home] = elo.Team(home, this_sim.date, utils.STARTING_ELOS[home])
        if away not in this_sim.teams: 
            this_sim.teams[away] = elo.Team(away, this_sim.date, utils.STARTING_ELOS[away])

        pre_home = this_sim.get_elo(home)
        pre_away = this_sim.get_elo(away)

        is_playoffs = True if row['Date'][5:7] in ['10', '11'] else False
        k_factor = params['k_factor'] if not is_playoffs else params['k_factor'] + elo.PLAYOFF_K_EXTRA
        margin_mult = 1 if not is_playoffs else elo.PLAYOFF_MARGIN_MULT

        #add the home advantage to whichever side is home
        Welo_0, Lelo_0 = this_sim.get_elo(winner), this_sim.get_elo(loser)
        if winner == home: Welo_0 += params['home_advantage']
        else: Lelo_0 += params['home_advantage']
        
        elo_margin = (Welo_0 - Lelo_0) * margin_mult #winner minus loser elo multipled by extra margin if playoffs
        w_winp = 1 / (1 + 10**(-elo_margin/400))
        this_sim.track_prediction(w_winp)

        MoV = winnerScore - loserScore #how much did winner win by
        MoV_multiplier = elo.calc_MoV_multiplier(elo_margin, MoV)
        elo_delta = k_factor * MoV_multiplier * (1 - w_winp)
        this_sim.update_elos(winner, loser, elo_delta)

    # Calculate performance metrics
    pred_accuracy = np.mean([1 if p > 0.5  else 0 for p in this_sim.prediction_results])
    brier_score = np.mean([p ** 2 for p in this_sim.prediction_results])

    return {
        'prediction_accuracy': pred_accuracy,
        'brier_score': brier_score
    }

def grid_search():
    """
    Perform grid search over parameter combinations
    """
    # Load historical data
    data = pd.read_csv(utils.get_latest_data_filepath())
    
    # Define parameter grid
    param_grid = {
        'k_factor': [1.5, 2, 2.5, 3, 3.5, 4, 4.5],
        'home_advantage': [10, 11, 12, 13, 14, 15, 16],
        'season_carry': [0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68],
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
    results_df.to_csv(f'tuning_results_{timestamp}.csv', index=False)

if __name__ == '__main__':
    grid_search() 