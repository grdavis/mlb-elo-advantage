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

This scrapes Baseball Reference + ScoresAndOdds, runs the Elo engine on historical data, and outputs:
- CSV: `OUTPUTS/Game Predictions Based on Ratings through YYYY-MM-DD.csv`
- Markdown: `docs/index.md`
- Data: `DATA/game_log_YYYY-MM-DD.csv`

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
```

### Known caveats

- **Plotly browser output**: `utils.table_output()` tries to open Plotly tables in a browser. In headless environments this produces harmless dbus/GPU errors in stderr but does not affect pipeline correctness (caught by try/except).
- **Scraping latency**: The odds scraper calls ScoresAndOdds for each date individually; during a long catchup (e.g. after multi-day gap) this can be slow.
- **DATA directory**: The pipeline expects at least one `DATA/game_log_*.csv` file to exist as the baseline. The repo ships with these files already.
