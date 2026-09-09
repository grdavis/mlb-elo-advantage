# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

MLB Elo Advantage is a pure Python data pipeline that scrapes MLB game results and betting odds, computes Elo ratings, and generates daily predictions. There is no database, no frontend, and no Docker. See `README.md` for full details.

### Running the application

Activate the virtual environment, then run the main pipeline:

```bash
source .venv/bin/activate
python predictions.py
```

This scrapes Baseball Reference + ScoresAndOdds, refreshes probable pitchers and
starter game logs from the MLB Stats API (`statsapi.mlb.com`), runs the Elo
engine, applies a rolling Game Score adjustment, and outputs:
- CSV: `OUTPUTS/Game Predictions Based on Ratings through YYYY-MM-DD.csv`
- Markdown: `docs/index.md`
- Data: `DATA/game_log_YYYY-MM-DD.csv`
- Pitcher cache (committed): `DATA/pitchers/game_starters.csv`, `DATA/pitchers/pitcher_game_logs.csv`

The full pipeline takes ~10 minutes due to network scraping + Monte Carlo season simulations (~12k–50k iterations).

### Running without network (fast, offline)

To run just the Elo simulation on existing data (no scraping, ~1 second):

```bash
source .venv/bin/activate
python elo.py
```

### Linting and testing

This project has no formal linting configuration or test suite. You can check for syntax errors with:

```bash
source .venv/bin/activate
python -m py_compile predictions.py
python -m py_compile elo.py
python -m py_compile scraper.py
python -m py_compile utils.py
python -m py_compile season_predictions.py
python -m py_compile pitcher_model.py
```

### Known caveats

- **Plotly browser output**: `utils.table_output()` tries to open Plotly tables in a browser. In headless environments this produces harmless dbus/GPU errors in stderr but does not affect pipeline correctness (caught by try/except).
- **Scraping latency**: The odds scraper calls ScoresAndOdds for each date individually; during a long catchup (e.g. after multi-day gap) this can be slow.
- **DATA directory**: The pipeline expects at least one `DATA/game_log_*.csv` file and the pitcher cache under `DATA/pitchers/` (shipped in the repo). Daily runs update those files. `python pitcher_model.py` rebuilds the pitcher cache from scratch (`force=True` path).
- **MLB Stats API**: `predictions.py` calls `pitcher_model.refresh_pitcher_cache()` before publishing. That replaces the current calendar year’s starter list and refreshes logs for pitchers on games dated `today-16 days` and later (so remaining-season listed probables are included). If the request fails, the job logs the error and falls back to the committed cache (missing starters are treated as Game Score 50).
- **Pitcher cache vs ratings**: `DATA/pitchers/game_starters.csv` and `pitcher_game_logs.csv` are committed (the daily Action uses `git add -A`). Rolling Game Score is derived and gitignored. Team Elo in `elo.py` still updates from results only; `pitcher_adj` is prediction-time / rest-of-season sim only.
- **Published tables**: `docs/index.md` and the prediction CSV include Away/Home pitcher names. Blank names are unlisted probables, not a join failure; those sides use GS 50.
