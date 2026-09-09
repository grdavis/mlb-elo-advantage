"""
Starting-pitcher adjustments for pre-game win probabilities.

Team Elo is a season-long rating and does not know who is on the mound.
FiveThirtyEight's betting-relevant MLB model added a rolling Game Score
adjustment for each starter; this module reconstructs that idea from the
MLB Stats API so predictions are not systematically faded by the market
when an ace faces a replacement-level opponent.

All rolling features are strictly causal (prior starts only).
"""
from __future__ import annotations
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

import utils

CACHE_DIR = os.path.join(utils.DATA_FOLDER, 'pitchers')
STARTERS_PATH = os.path.join(CACHE_DIR, 'game_starters.csv')
LOGS_PATH = os.path.join(CACHE_DIR, 'pitcher_game_logs.csv')
ROLLING_PATH = os.path.join(CACHE_DIR, 'pitcher_rolling.csv')

SESSION_HEADERS = {'User-Agent': 'mlb-elo-advantage/1.0 (research)'}

# MLB schedule abbreviations -> repo team codes
MLB_ABBR_TO_REPO = {
    'LAA': 'ANA', 'ANA': 'ANA',
    'ARI': 'ARI', 'AZ': 'ARI',
    'ATL': 'ATL',
    'BAL': 'BAL',
    'BOS': 'BOS',
    'CHC': 'CHC',
    'CWS': 'CHW', 'CHW': 'CHW',
    'CIN': 'CIN',
    'CLE': 'CLE',
    'COL': 'COL',
    'DET': 'DET',
    'MIA': 'FLA', 'FLA': 'FLA',
    'HOU': 'HOU',
    'KC': 'KCR', 'KCR': 'KCR',
    'LAD': 'LAD', 'LA': 'LAD',
    'MIL': 'MIL',
    'MIN': 'MIN',
    'NYM': 'NYM',
    'NYY': 'NYY',
    'OAK': 'OAK', 'ATH': 'OAK',
    'PHI': 'PHI',
    'PIT': 'PIT',
    'SD': 'SDP', 'SDP': 'SDP',
    'SEA': 'SEA',
    'SF': 'SFG', 'SFG': 'SFG',
    'STL': 'STL',
    'TB': 'TBD', 'TBR': 'TBD', 'TBD': 'TBD',
    'TEX': 'TEX',
    'TOR': 'TOR',
    'WSH': 'WSN', 'WAS': 'WSN', 'WSN': 'WSN',
}

# Fallback by full MLB team name (schedule payloads often omit abbreviation)
MLB_NAME_TO_REPO = {
    'Los Angeles Angels': 'ANA',
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN',
    'Cleveland Indians': 'CLE',
    'Cleveland Guardians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Miami Marlins': 'FLA',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KCR',
    'Los Angeles Dodgers': 'LAD',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'OAK',
    'Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',
    'Seattle Mariners': 'SEA',
    'San Francisco Giants': 'SFG',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBD',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSN',
}

# Elo points per Game Score point above/below 50. ~1.4 maps a 10-GS gap to
# ~20 Elo (~3% win probability), in line with published 538 pitcher_adj ranges.
DEFAULT_GS_TO_ELO = 1.4
ROLLING_SPAN = 8  # EWM span in starts
MIN_STARTS_FOR_FULL_WEIGHT = 4


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(SESSION_HEADERS)
    return s


def _mlb_team_to_repo(team_obj: dict) -> str | None:
    abbr = (team_obj.get('abbreviation') or team_obj.get('fileCode') or '').upper()
    if abbr in MLB_ABBR_TO_REPO:
        return MLB_ABBR_TO_REPO[abbr]
    name = team_obj.get('name') or ''
    return MLB_NAME_TO_REPO.get(name)


def fetch_season_starters(season: int, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or _session()
    url = (
        'https://statsapi.mlb.com/api/v1/schedule'
        f'?sportId=1&season={season}&gameType=R,F,D,L,W&hydrate=probablePitcher,team'
    )
    r = s.get(url, timeout=90)
    r.raise_for_status()
    rows = []
    for day in r.json().get('dates', []):
        for g in day.get('games', []):
            home = g['teams']['home']
            away = g['teams']['away']
            ht = _mlb_team_to_repo(home.get('team') or {})
            at = _mlb_team_to_repo(away.get('team') or {})
            if not ht or not at:
                continue
            hp = home.get('probablePitcher') or {}
            ap = away.get('probablePitcher') or {}
            rows.append({
                'Date': (g.get('officialDate') or day.get('date') or '')[:10],
                'Home': ht,
                'Away': at,
                'gamePk': g.get('gamePk'),
                'gameNumber': g.get('gameNumber', 1),
                'home_pitcher_id': hp.get('id'),
                'home_pitcher': hp.get('fullName'),
                'away_pitcher_id': ap.get('id'),
                'away_pitcher': ap.get('fullName'),
            })
    return pd.DataFrame(rows)


def fetch_all_starters(start_season: int = 2017, end_season: int = None,
                       force: bool = False) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(STARTERS_PATH) and not force:
        return pd.read_csv(STARTERS_PATH)
    if end_season is None:
        end_season = datetime.today().year
    s = _session()
    frames = []
    for season in range(start_season, end_season + 1):
        print(f'Fetching MLB starters for {season}...')
        frames.append(fetch_season_starters(season, s))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(STARTERS_PATH, index=False)
    print(f'Wrote {STARTERS_PATH} ({len(out)} games)')
    return out


def _parse_ip(ip) -> float:
    if ip is None or (isinstance(ip, float) and np.isnan(ip)):
        return 0.0
    s = str(ip)
    if '.' in s:
        whole, frac = s.split('.', 1)
        return float(whole or 0) + int(frac or 0) / 3.0
    return float(s or 0)


def bill_james_game_score(stat: dict) -> float:
    """Tango/FanGraphs-style Game Score using MLB box stats."""
    outs = int(stat.get('outs') or 0)
    k = int(stat.get('strikeOuts') or 0)
    bb = int(stat.get('baseOnBalls') or 0)
    h = int(stat.get('hits') or 0)
    er = int(stat.get('earnedRuns') or 0)
    hr = int(stat.get('homeRuns') or 0)
    return 50 + outs + k - 2 * bb - 2 * h - 3 * er - 6 * hr


def fetch_pitcher_logs(pitcher_id: int, session: requests.Session,
                       start: str = '2016-01-01', end: str = None) -> pd.DataFrame:
    if end is None:
        end = f'{datetime.today().year}-12-31'
    url = (
        f'https://statsapi.mlb.com/api/v1/people/{int(pitcher_id)}/stats'
        f'?stats=gameLog&group=pitching&startDate={start}&endDate={end}'
    )
    for attempt in range(4):
        try:
            r = session.get(url, timeout=45)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            stats = r.json().get('stats') or []
            splits = stats[0].get('splits', []) if stats else []
            rows = []
            for sp in splits:
                st = sp.get('stat') or {}
                if int(st.get('gamesStarted') or 0) < 1:
                    continue
                rows.append({
                    'pitcher_id': int(pitcher_id),
                    'pitcher': (sp.get('player') or {}).get('fullName'),
                    'Date': sp.get('date'),
                    'gamePk': (sp.get('game') or {}).get('gamePk'),
                    'outs': int(st.get('outs') or 0),
                    'strikeOuts': int(st.get('strikeOuts') or 0),
                    'baseOnBalls': int(st.get('baseOnBalls') or 0),
                    'hits': int(st.get('hits') or 0),
                    'earnedRuns': int(st.get('earnedRuns') or 0),
                    'homeRuns': int(st.get('homeRuns') or 0),
                    'inningsPitched': _parse_ip(st.get('inningsPitched')),
                    'game_score': bill_james_game_score(st),
                })
            return pd.DataFrame(rows)
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return pd.DataFrame()


def fetch_all_pitcher_logs(pitcher_ids, force: bool = False, max_workers: int = 12) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(LOGS_PATH) and not force:
        return pd.read_csv(LOGS_PATH)
    ids = sorted({int(i) for i in pitcher_ids if pd.notna(i)})
    print(f'Fetching game logs for {len(ids)} pitchers...')
    frames = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # thread-local sessions via one session per worker is hard; use a few sessions
        futs = {}
        sessions = [_session() for _ in range(max_workers)]
        for n, pid in enumerate(ids):
            fut = ex.submit(fetch_pitcher_logs, pid, sessions[n % max_workers])
            futs[fut] = pid
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None and len(df):
                frames.append(df)
            done += 1
            if done % 100 == 0:
                print(f'  {done}/{len(ids)} pitchers')
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(LOGS_PATH, index=False)
    print(f'Wrote {LOGS_PATH} ({len(out)} starts)')
    return out


def build_rolling_gs(logs: pd.DataFrame, span: int = ROLLING_SPAN) -> pd.DataFrame:
    """Causal rolling Game Score.

    rolling_gs: pre-game rating for a start on that date (prior starts only).
    gs_after: rating after the start, used as the as-of rating for later games
    (including future slates where the pitcher has no row on the game date).
    """
    if logs.empty:
        return logs
    logs = logs.sort_values(['pitcher_id', 'Date']).copy()
    parts = []
    for pid, g in logs.groupby('pitcher_id', sort=False):
        g = g.sort_values('Date').copy()
        g['starts_prior'] = np.arange(len(g))
        prior = g['game_score'].shift(1)
        g['rolling_gs'] = prior.ewm(span=span, min_periods=1).mean()
        w_pre = np.minimum(g['starts_prior'] / float(MIN_STARTS_FOR_FULL_WEIGHT), 1.0)
        g['rolling_gs'] = 50.0 * (1 - w_pre) + g['rolling_gs'].fillna(50.0) * w_pre
        incl = g['game_score'].ewm(span=span, min_periods=1).mean()
        w_post = np.minimum((g['starts_prior'] + 1) / float(MIN_STARTS_FOR_FULL_WEIGHT), 1.0)
        g['gs_after'] = 50.0 * (1 - w_post) + incl * w_post
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def ensure_pitcher_cache(force: bool = False) -> tuple:
    starters = fetch_all_starters(force=force)
    ids = set(starters['home_pitcher_id'].dropna().tolist()) | set(starters['away_pitcher_id'].dropna().tolist())
    logs = fetch_all_pitcher_logs(ids, force=force)
    rolling = build_rolling_gs(logs)
    return starters, rolling


def _replace_pitcher_logs(logs: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    if updates is None or len(updates) == 0:
        return logs
    if logs is None or len(logs) == 0:
        return updates
    keep = ~logs['pitcher_id'].isin(set(updates['pitcher_id'].unique()))
    return pd.concat([logs.loc[keep], updates], ignore_index=True)


def refresh_pitcher_cache(lookback_days: int = 16) -> tuple:
    """Refresh current-season probables and recent starter game logs (daily pipeline).

    Replaces the current calendar year in game_starters.csv, then re-fetches
    career-window logs for every pitcher listed on a game dated
    (today - lookback_days) or later. Future remaining-season dates are
    therefore included; pitchers who are not on that list keep their last
    committed logs. Rolling GS is rebuilt in memory (not required on disk).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    today = datetime.today()
    year = today.year
    print(f'Refreshing {year} probable pitchers from MLB Stats API...')
    season_st = fetch_season_starters(year)
    if os.path.exists(STARTERS_PATH):
        old = pd.read_csv(STARTERS_PATH)
        old = old.loc[~old['Date'].astype(str).str.startswith(str(year))]
        starters = pd.concat([old, season_st], ignore_index=True)
    else:
        starters = season_st
    starters.to_csv(STARTERS_PATH, index=False)

    cutoff = (today - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    recent = season_st.loc[season_st['Date'].astype(str) >= cutoff]
    ids = set(recent['home_pitcher_id'].dropna().tolist()) | set(recent['away_pitcher_id'].dropna().tolist())
    # always include today's listed starters even if lookback slice is empty
    today_s = utils.date_to_string(today)
    today_rows = season_st.loc[season_st['Date'].astype(str) == today_s]
    ids |= set(today_rows['home_pitcher_id'].dropna().tolist()) | set(today_rows['away_pitcher_id'].dropna().tolist())
    ids = {int(i) for i in ids}

    logs = pd.read_csv(LOGS_PATH) if os.path.exists(LOGS_PATH) else pd.DataFrame()
    if ids:
        print(f'Refreshing game logs for {len(ids)} recent/today starters...')
        session = _session()
        frames = [fetch_pitcher_logs(pid, session) for pid in sorted(ids)]
        updates = pd.concat([f for f in frames if f is not None and len(f)], ignore_index=True) if frames else pd.DataFrame()
        logs = _replace_pitcher_logs(logs, updates)
        if len(logs):
            logs.to_csv(LOGS_PATH, index=False)

    rolling = build_rolling_gs(logs) if logs is not None and len(logs) else pd.DataFrame()
    return starters, rolling


def _dedupe_starters(starters: pd.DataFrame) -> pd.DataFrame:
    s = starters.copy()
    s['matchup_on_date'] = s.groupby(['Date', 'Home', 'Away']).cumcount() + 1
    return s


def _asof_pitcher_rating(games: pd.DataFrame, pid_col: str, rolling: pd.DataFrame) -> pd.Series:
    """Last post-start GS strictly before the game date (no same-day leakage)."""
    left = pd.DataFrame({
        '_row': np.arange(len(games)),
        'Date': pd.to_datetime(games['Date']),
        pid_col: games[pid_col].values,
    })
    right = rolling[['pitcher_id', 'Date', 'gs_after']].copy()
    right['Date'] = pd.to_datetime(right['Date'])
    right[pid_col] = pd.to_numeric(right['pitcher_id'], errors='coerce')
    right = right.rename(columns={'Date': 'pdate', 'gs_after': '_gs'})
    out = np.full(len(games), np.nan)
    for pid, lg in left.groupby(pid_col, sort=False):
        if pd.isna(pid):
            continue
        rg = right.loc[right[pid_col] == pid, ['pdate', '_gs']].sort_values('pdate')
        if rg.empty:
            continue
        lg = lg.sort_values('Date')
        m = pd.merge_asof(
            lg, rg, left_on='Date', right_on='pdate',
            direction='backward', allow_exact_matches=False,
        )
        out[m['_row'].to_numpy()] = m['_gs'].to_numpy()
    return pd.Series(out, index=games.index)


def attach_pitchers_to_games(games: pd.DataFrame, starters: pd.DataFrame | None = None,
                             rolling: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join starter IDs and pre-game rolling GS onto a game log / combo frame."""
    if starters is None or rolling is None:
        starters, rolling = ensure_pitcher_cache()
    games = games.copy()
    games['Date'] = games['Date'].astype(str).str[:10]
    games['matchup_on_date'] = games.groupby(['Date', 'Home', 'Away']).cumcount() + 1
    st = _dedupe_starters(starters)
    st['Date'] = st['Date'].astype(str).str[:10]
    st['home_pitcher_id'] = pd.to_numeric(st['home_pitcher_id'], errors='coerce')
    st['away_pitcher_id'] = pd.to_numeric(st['away_pitcher_id'], errors='coerce')
    merged = games.merge(
        st[['Date', 'Home', 'Away', 'matchup_on_date',
            'home_pitcher_id', 'home_pitcher', 'away_pitcher_id', 'away_pitcher']],
        on=['Date', 'Home', 'Away', 'matchup_on_date'],
        how='left',
    )
    for col in ['home_pitcher_id', 'away_pitcher_id']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
    roll = rolling.copy()
    roll['Date'] = roll['Date'].astype(str).str[:10]
    roll['pitcher_id'] = pd.to_numeric(roll['pitcher_id'], errors='coerce')
    merged['home_rolling_gs'] = _asof_pitcher_rating(merged, 'home_pitcher_id', roll).fillna(50.0)
    merged['away_rolling_gs'] = _asof_pitcher_rating(merged, 'away_pitcher_id', roll).fillna(50.0)
    return merged


def pitcher_adj_lookup(games: pd.DataFrame, starters=None, rolling=None, scale=DEFAULT_GS_TO_ELO) -> dict:
    """Map (Date, Home, Away, matchup_on_date) -> net home-side Elo from starters."""
    attached = attach_pitchers_to_games(games, starters, rolling)
    lookup = {}
    for _, r in attached.iterrows():
        key = (str(r['Date'])[:10], r['Home'], r['Away'], int(r['matchup_on_date']))
        lookup[key] = pitcher_elo_adj(r['home_rolling_gs'], r['away_rolling_gs'], scale)
    return lookup


def pitcher_elo_adj(home_gs, away_gs, scale=DEFAULT_GS_TO_ELO):
    """Net Elo added to the home side from starter quality (home GS - away GS)."""
    return (float(home_gs) - float(away_gs)) * float(scale)


def adjust_home_winp(home_pre_prob: float, home_gs: float, away_gs: float,
                     scale: float = DEFAULT_GS_TO_ELO) -> float:
    """Shift an Elo home-win probability by the starter Game Score gap."""
    # invert logistic to Elo margin, add pitcher adj, re-logistic
    # p = 1/(1+10^(-m/400)) => m = 400*log10(p/(1-p))
    p = min(max(float(home_pre_prob), 1e-6), 1 - 1e-6)
    margin = 400.0 * np.log10(p / (1 - p))
    margin += pitcher_elo_adj(home_gs, away_gs, scale)
    return float(1 / (1 + 10 ** (-margin / 400.0)))


def apply_pitcher_probs(df: pd.DataFrame, scale: float = DEFAULT_GS_TO_ELO) -> pd.DataFrame:
    out = df.copy()
    adj_p = [
        adjust_home_winp(hp, hgs, ags, scale)
        for hp, hgs, ags in zip(out['HOME_PRE_PROB'], out['home_rolling_gs'], out['away_rolling_gs'])
    ]
    out['HOME_PRE_PROB_RAW'] = out['HOME_PRE_PROB']
    out['AWAY_PRE_PROB_RAW'] = out['AWAY_PRE_PROB']
    out['HOME_PRE_PROB'] = adj_p
    out['AWAY_PRE_PROB'] = 1.0 - out['HOME_PRE_PROB']
    out['pitcher_elo_adj'] = (out['home_rolling_gs'] - out['away_rolling_gs']) * scale
    return out


if __name__ == '__main__':
    ensure_pitcher_cache()
