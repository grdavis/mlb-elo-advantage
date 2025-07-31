#!/usr/bin/env python3
"""
Historical MLB Data Scraper
Scrapes historical MLB game data from Action Network API and formats it to match existing CSV structure.
"""

import requests
import json
import csv
import time
import random
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from bs4 import BeautifulSoup
import os # Added for run_single_season_matching

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLB Season Date Ranges (Regular Season Only)
# Format: (start_date, end_date) as datetime objects
# Source: MLB official schedules and Wikipedia
MLB_SEASON_DATES = {
    2016: (datetime(2016, 4, 3), datetime(2016, 11, 2)),   # From Wikipedia: https://en.wikipedia.org/wiki/2016_Major_League_Baseball_season
    2017: (datetime(2017, 4, 2), datetime(2017, 11, 1)),   # From Wikipedia: https://en.wikipedia.org/wiki/2017_Major_League_Baseball_season
    2018: (datetime(2018, 3, 29), datetime(2018, 10, 28)), # From Wikipedia: https://en.wikipedia.org/wiki/2018_Major_League_Baseball_season
    2019: (datetime(2019, 3, 20), datetime(2019, 10, 30)), # From Wikipedia: https://en.wikipedia.org/wiki/2019_Major_League_Baseball_season
    2020: (datetime(2020, 7, 23), datetime(2020, 10, 27)), # COVID-shortened season (60 games) - From Wikipedia: https://en.wikipedia.org/wiki/2020_Major_League_Baseball_season
    2021: (datetime(2021, 4, 1), datetime(2021, 11, 2)),   # From Wikipedia: https://en.wikipedia.org/wiki/2021_Major_League_Baseball_season
    2022: (datetime(2022, 4, 7), datetime(2022, 11, 5))  # From Wikipedia: https://en.wikipedia.org/wiki/2022_Major_League_Baseball_season
}

# Import functions from utils.py
from utils import BR_TEAM_MAP, date_to_string, string_to_date

class ActionNetworkScraper:
    """Scraper for Action Network MLB API with anti-detection measures"""
    
    def __init__(self, max_requests_per_minute=30, max_requests_per_hour=300):
        self.base_url = "https://api.actionnetwork.com/web/v2/scoreboard/mlb"
        self.session = requests.Session()
        
        # Rate limiting settings
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.request_times = []  # Track request timestamps
        
        # Add headers to mimic a browser request
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.actionnetwork.com/',
            'Origin': 'https://www.actionnetwork.com',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })
        
        # Team name mapping (full names to abbreviations)
        self.team_mapping = {
            'Yankees': 'NYY', 'Phillies': 'PHI', 'Royals': 'KCR', 'Guardians': 'CLE', 'Tigers': 'DET',
            'Blue Jays': 'TOR', 'Reds': 'CIN', 'Rays': 'TBD', 'Pirates': 'PIT', 'Diamondbacks': 'ARI',
            'Orioles': 'BAL', 'Rockies': 'COL', 'Rangers': 'TEX', 'Braves': 'ATL', 'Brewers': 'MIL',
            'Marlins': 'FLA', 'White Sox': 'CHW', 'Cubs': 'CHC', 'Astros': 'HOU', 'Athletics': 'OAK',
            'Twins': 'MIN', 'Nationals': 'WSN', 'Red Sox': 'BOS', 'Dodgers': 'LAD', 'Cardinals': 'STL',
            'Padres': 'SDP', 'Giants': 'SFG', 'Mets': 'NYM', 'Angels': 'ANA', 'Mariners': 'SEA'
        }

    def is_in_season(self, date_str: str) -> bool:
        """
        Check if a given date (YYYYMMDD format) falls within any MLB regular season.
        
        Args:
            date_str: Date in YYYYMMDD format
            
        Returns:
            bool: True if date is during regular season, False otherwise
        """
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            year = date_obj.year
            
            # Check if we have season data for this year
            if year not in MLB_SEASON_DATES:
                logger.warning(f"No season data for year {year}, assuming off-season")
                return False
            
            start_date, end_date = MLB_SEASON_DATES[year]
            
            # Check if date falls within the season
            return start_date <= date_obj <= end_date
            
        except ValueError as e:
            logger.error(f"Invalid date format {date_str}: {e}")
            return False

    def filter_season_dates(self, date_list: List[str]) -> List[str]:
        """
        Filter a list of dates to only include those during MLB regular seasons.
        
        Args:
            date_list: List of dates in YYYYMMDD format
            
        Returns:
            List[str]: Filtered list containing only in-season dates
        """
        original_count = len(date_list)
        filtered_dates = [date for date in date_list if self.is_in_season(date)]
        filtered_count = len(filtered_dates)
        
        if filtered_count < original_count:
            logger.debug(f"Filtered out {original_count - filtered_count} off-season dates. "
                       f"Proceeding with {filtered_count} in-season dates.")
        
        return filtered_dates
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        
        # Clean old timestamps (older than 1 hour)
        self.request_times = [t for t in self.request_times if now - t < 3600]
        
        # Check minute limit
        requests_last_minute = len([t for t in self.request_times if now - t < 60])
        if requests_last_minute >= self.max_requests_per_minute:
            logger.warning(f"Rate limit reached: {requests_last_minute} requests in last minute")
            return False
        
        # Check hour limit
        if len(self.request_times) >= self.max_requests_per_hour:
            logger.warning(f"Hourly rate limit reached: {len(self.request_times)} requests in last hour")
            return False
        
        return True
    
    def _smart_delay(self, base_delay=2.0, jitter=0.5):
        """Add intelligent delays with randomization"""
        if not self._check_rate_limits():
            # If we're approaching limits, add extra delay
            extra_delay = random.uniform(30, 60)
            logger.info(f"Rate limit approaching, adding {extra_delay:.1f}s delay")
            time.sleep(extra_delay)
        
        # Add randomized delay to avoid patterns
        delay = base_delay + random.uniform(-jitter, jitter)
        delay = max(0.5, delay)  # Minimum 0.5 second delay
        
        logger.debug(f"Delaying for {delay:.2f} seconds")
        time.sleep(delay)
    
    def _make_request_with_retry(self, url: str, max_retries=3) -> Optional[Dict[str, Any]]:
        """Make HTTP request with retry logic and anti-detection measures"""
        for attempt in range(max_retries):
            try:
                # Check rate limits before making request
                if not self._check_rate_limits():
                    logger.warning("Rate limit exceeded, waiting...")
                    time.sleep(60)  # Wait 1 minute
                    continue
                
                # Add request timestamp
                self.request_times.append(time.time())
                
                # Add random delay between requests
                self._smart_delay()
                
                # Make the request
                response = self.session.get(url, timeout=15)
                
                # Check for rate limiting or blocking
                if response.status_code == 429:  # Too Many Requests
                    logger.warning("Rate limited by server (429), waiting 5 minutes...")
                    time.sleep(300)
                    continue
                elif response.status_code == 403:  # Forbidden
                    logger.error("Access forbidden (403) - possible IP blocking")
                    return None
                elif response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code}, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(random.uniform(5, 15))
                    continue
                
                # Success
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(5, 15))
                continue
        
        logger.error(f"Failed to make request after {max_retries} attempts")
        return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics for monitoring"""
        now = time.time()
        
        # Clean old timestamps
        self.request_times = [t for t in self.request_times if now - t < 3600]
        
        requests_last_minute = len([t for t in self.request_times if now - t < 60])
        requests_last_hour = len(self.request_times)
        
        return {
            'requests_last_minute': requests_last_minute,
            'requests_last_hour': requests_last_hour,
            'max_per_minute': self.max_requests_per_minute,
            'max_per_hour': self.max_requests_per_hour,
            'minute_limit_remaining': max(0, self.max_requests_per_minute - requests_last_minute),
            'hour_limit_remaining': max(0, self.max_requests_per_hour - requests_last_hour)
        }
    
    def print_usage_stats(self):
        """Print current usage statistics"""
        stats = self.get_usage_stats()
        logger.info(f"Usage Stats - Last minute: {stats['requests_last_minute']}/{stats['max_per_minute']}, "
                   f"Last hour: {stats['requests_last_hour']}/{stats['max_per_hour']}")

    def get_games_for_date(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Fetch games for a specific date"""
        url = f"{self.base_url}?date={date_str}"
        logger.info(f"Fetching data for {date_str}")
        
        try:
            data = self._make_request_with_retry(url)
            if data:
                logger.info("Successfully retrieved data")
                return data
            else:
                logger.error("Failed to retrieve data")
                return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {date_str}: {e}")
            return None

    def extract_game_data(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant game data from API response"""
        try:
            logger.debug(f"Game keys: {list(game.keys())}")
            logger.debug(f"Game ID: {game.get('id')}")
            
            # Extract team IDs
            away_team_id = game.get('away_team_id')
            home_team_id = game.get('home_team_id')
            
            # Extract team names from teams array
            teams = game.get('teams', [])
            away_team = 'Unknown'
            home_team = 'Unknown'
            
            for team in teams:
                team_id = team.get('id')
                if team_id == away_team_id:
                    full_name = team.get('display_name', 'Unknown')
                    away_team = self.team_mapping.get(full_name, full_name)
                elif team_id == home_team_id:
                    full_name = team.get('display_name', 'Unknown')
                    home_team = self.team_mapping.get(full_name, full_name)
            
            # Extract scores from boxscore
            boxscore = game.get('boxscore', {})
            stats = boxscore.get('stats', {})
            
            away_score = stats.get('away', {}).get('runs', 0)
            home_score = stats.get('home', {}).get('runs', 0)
            
            # Extract odds data from markets or latest_odds
            markets = game.get('markets', {})
            latest_odds = boxscore.get('latest_odds', {})
            
            logger.debug(f"Available books: {list(markets.keys())}")
            logger.debug(f"Latest odds available: {bool(latest_odds)}")
            
            # Try to get odds from any available bookmaker
            ou_line = None
            home_ml = None
            away_ml = None
            used_source = None
            
            # First, try to get odds from latest_odds (prioritize this as it's more current)
            if latest_odds:
                game_odds = latest_odds.get('game', {})
                
                if game_odds:
                    # Extract total (over/under) from latest_odds
                    if 'total' in game_odds:
                        ou_line = game_odds.get('total')
                    
                    # Extract moneyline from latest_odds
                    if 'ml_home' in game_odds:
                        home_ml = game_odds.get('ml_home')
                    if 'ml_away' in game_odds:
                        away_ml = game_odds.get('ml_away')
                    
                    if ou_line is not None and home_ml is not None and away_ml is not None:
                        used_source = "latest_odds"
                        logger.debug(f"Using odds from latest_odds")
            
            # If no odds found in latest_odds, try markets section as fallback
            if (ou_line is None or home_ml is None or away_ml is None) and markets:
                for book_id, book_data in markets.items():
                    book_markets = book_data.get('event', {})
                    
                    # Check if this book has the markets we need
                    total_markets = book_markets.get('total', [])
                    moneyline_markets = book_markets.get('moneyline', [])
                    
                    # Only use this book if it has both total and moneyline data
                    if total_markets and moneyline_markets:
                        # Extract total (over/under)
                        ou_line = total_markets[0].get('value')
                        
                        # Extract moneyline
                        for ml in moneyline_markets:
                            if ml.get('side') == 'home':
                                home_ml = ml.get('odds')
                            elif ml.get('side') == 'away':
                                away_ml = ml.get('odds')
                        
                        # If we found valid odds, break out of the loop
                        if ou_line is not None and home_ml is not None and away_ml is not None:
                            used_source = f"markets_{book_id}"
                            logger.debug(f"Using odds from markets book {book_id}")
                            break
            
            logger.debug(f"Used source: {used_source}, Extracted odds - OU: {ou_line}, Home ML: {home_ml}, Away ML: {away_ml}")
            
            return {
                'away_team': away_team,
                'home_team': home_team,
                'away_score': away_score,
                'home_score': home_score,
                'ou_line': ou_line,
                'home_ml': home_ml,
                'away_ml': away_ml
            }
        except Exception as e:
            logger.error(f"Error extracting game data: {e}")
            return None

    def scrape_single_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Scrape games for a single date"""
        # Check if date is in season
        if not self.is_in_season(date_str):
            return []
        
        try:
            data = self.get_games_for_date(date_str)
            if not data:
                return []
            
            # Extract games from the response
            games = data.get('games', [])
            
            # Format date from YYYYMMDD to YYYY-MM-DD
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            extracted_games = []
            for game in games:
                game_data = self.extract_game_data(game)
                if game_data:
                    # Add the formatted date to the game data
                    game_data['date'] = formatted_date
                    extracted_games.append(game_data)
            
            return extracted_games
            
        except Exception as e:
            logger.error(f"Unexpected error for {date_str}: {e}")
            return []

    def scrape_date_range(self, start_date: str, end_date: str, output_file: str) -> None:
        """Scrape games for a date range and save to CSV"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_games = []
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y%m%d')
            
            data = self.get_games_for_date(date_str)
            if data and 'games' in data:
                games = data['games']
                logger.info(f"Found {len(games)} games for {date_str}")
                
                for game in games:
                    game_data = self.extract_game_data(game)
                    if game_data:
                        all_games.append(game_data)
            
            # Rate limiting
            time.sleep(1)
            current_date += timedelta(days=1)
        
        # Save to CSV
        self.save_to_csv(all_games, output_file)
        logger.info(f"Saved {len(all_games)} games to {output_file}")

    def scrape_multiple_dates(self, date_list: List[str], output_file: str) -> None:
        """Scrape games for multiple dates and save to a single CSV file with mid-season contingencies"""
        # Filter out off-season dates
        filtered_dates = self.filter_season_dates(date_list)
        
        if not filtered_dates:
            logger.warning("No in-season dates found in the provided date list")
            return
        
        all_games = []
        save_frequency = 10  # Save every 10 dates as a contingency
        
        for i, date_str in enumerate(filtered_dates):
            try:
                games = self.scrape_single_date(date_str)
                if games:
                    all_games.extend(games)
                    logger.info(f"Date {date_str}: {len(games)} games")
                else:
                    logger.info(f"Date {date_str}: 0 games")
                
                # Rate limiting is now handled automatically in _smart_delay()
                
                # Mid-season contingency save every N dates
                if (i + 1) % save_frequency == 0:
                    contingency_file = output_file.replace('.csv', f'_contingency_{i+1}_dates.csv')
                    self.save_to_csv(all_games, contingency_file)
                    logger.info(f"💾 Contingency save: {len(all_games)} games saved to {contingency_file}")
                
            except Exception as e:
                logger.error(f"❌ Error scraping date {date_str}: {e}")
                
                # Emergency save on error
                if all_games:
                    emergency_file = output_file.replace('.csv', f'_emergency_{date_str}.csv')
                    self.save_to_csv(all_games, emergency_file)
                    logger.info(f"🚨 Emergency save: {len(all_games)} games saved to {emergency_file}")
                
                # Continue with next date instead of failing completely
                logger.info("Continuing with next date...")
                continue
        
        # Final save
        if all_games:
            self.save_to_csv(all_games, output_file)
            logger.info(f"✅ Successfully scraped {len(all_games)} total games and saved to {output_file}")
        else:
            logger.warning("No games found for any of the specified dates")

    def save_to_csv(self, games: List[Dict[str, Any]], output_file: str) -> None:
        """Save games data to CSV file"""
        if not games:
            logger.warning("No games to save")
            return
            
        # Define the CSV columns to match the expected format
        fieldnames = ['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML']
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for game in games:
                    # Map the extracted data to CSV format with proper formatting
                    row = {
                        'Date': game['date'],
                        'Home': game['home_team'],
                        'Away': game['away_team'],
                        'Home_Score': f"{game['home_score']:.1f}",
                        'Away_Score': f"{game['away_score']:.1f}",
                        'OU_Line': f"{game['ou_line']:.1f}" if game['ou_line'] is not None else '',
                        'Home_ML': f"{game['home_ml']:.1f}" if game['home_ml'] is not None else '',
                        'Away_ML': f"{game['away_ml']:.1f}" if game['away_ml'] is not None else ''
                    }
                    writer.writerow(row)
                    
            logger.info(f"Saved {len(games)} games to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise

    def emergency_save(self, games: List[Dict[str, Any]], season: int, reason: str = "unknown") -> str:
        """Emergency save function for unexpected interruptions"""
        if not games:
            logger.warning("No games to save in emergency")
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emergency_file = f"DATA/emergency_season_{season}_{reason}_{timestamp}.csv"
        
        try:
            self.save_to_csv(games, emergency_file)
            logger.info(f"🚨 Emergency save completed: {len(games)} games saved to {emergency_file}")
            return emergency_file
        except Exception as e:
            logger.error(f"Failed emergency save: {e}")
            return ""

class BaseballReferenceScraper:
    """Scraper for Baseball Reference historical schedules"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def get_section_date(self, section, previous_date=None):
        """Extract date from a section header, handling 'Today's Games' edge case"""
        date_header = section.find('h3').text
        if date_header == "Today's Games":
            # Check if games have scores (indicating these aren't actually today's games)
            rows = section.find_all("p")
            has_scores = any(len(row.text.split('\n')[1:-2]) == 5 for row in rows)
            if has_scores:
                # If we have scores, this must be the day after the previous section
                if previous_date:
                    next_date = datetime.strptime(previous_date, '%Y-%m-%d') + timedelta(days=1)
                    return date_to_string(next_date)
                else:
                    return date_to_string(datetime.today())
            else:
                return date_to_string(datetime.today())  # No scores means these are actually today's games
        else:
            # Normal date header processing
            return date_to_string(datetime.strptime(date_header[date_header.find(',')+2:], '%B %d, %Y'))

    def scrape_season_schedule(self, season: int) -> pd.DataFrame:
        """
        Scrape the complete season schedule from Baseball Reference for a given year.
        
        Args:
            season: Year to scrape (e.g., 2021)
            
        Returns:
            pd.DataFrame: DataFrame with columns ['Date', 'Home', 'Away', 'Home_Score', 'Away_Score']
        """
        logger.info(f"Scraping Baseball Reference schedule for {season} season...")
        
        # Construct URL for the season
        schedule_url = f'https://www.baseball-reference.com/leagues/majors/{season}-schedule.shtml'
        
        try:
            response = self.session.get(schedule_url, timeout=30)
            response.raise_for_status()
            
            all_data = []
            data = response.content
            section_content = BeautifulSoup(data, 'html.parser').find_all('div', {'class': 'section_content'})
            
            # There should be one section_content for regular season and one for post-season
            previous_date = None
            
            for sc in section_content:
                sections = sc.find_all("div")
                for section in sections:
                    date = self.get_section_date(section, previous_date)
                    previous_date = date
                    
                    rows = section.find_all("p")
                    for row in rows:
                        this_row_contents = row.text
                        this_row_contents = this_row_contents.split('\n')[1:-2]
                        
                        if this_row_contents == [] or this_row_contents[1] == '(Spring)':
                            continue
                            
                        # Determine if this is playoffs (approximate date)
                        is_playoffs = date >= f'{season}-09-29'
                        
                        if len(this_row_contents) in [6, 7]:
                            # Future games, remove the postseason series designation for easier scraping if playoffs
                            if is_playoffs:
                                this_row_contents.pop(1)
                            all_data.append([date,
                                BR_TEAM_MAP[this_row_contents[4].strip(' ')],
                                BR_TEAM_MAP[this_row_contents[1]],
                                '', ''])
                        elif len(this_row_contents) == 5:
                            # Games already occurred
                            all_data.append([date,
                                BR_TEAM_MAP[this_row_contents[3].strip(' ')],
                                BR_TEAM_MAP[this_row_contents[0].strip(' ')],
                                this_row_contents[4].strip(' ').strip('(').strip(')'),
                                this_row_contents[1].strip(' ').strip('(').strip(')')
                                ])

            df = pd.DataFrame(all_data, columns=['Date', 'Home', 'Away', 'Home_Score', 'Away_Score'])
            logger.info(f"Successfully scraped {len(df)} games for {season} season")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping {season} schedule: {e}")
            return pd.DataFrame(columns=['Date', 'Home', 'Away', 'Home_Score', 'Away_Score'])

def match_golden_schedule_with_odds(season: int, odds_file: str = None, output_file: str = None) -> pd.DataFrame:
    """
    Match Baseball Reference "golden" schedule with historical odds data.
    Performs a left join to keep all games from BR schedule and enrich with odds where available.
    
    Args:
        season: Year to process (e.g., 2021)
        odds_file: Path to historical odds CSV file (if None, will construct default path)
        output_file: Path to save matched data (if None, will construct default path)
        
    Returns:
        pd.DataFrame: Matched data with all BR games and odds data where available
    """
    logger.info(f"=== Matching Golden Schedule with Odds for {season} ===")
    
    # Construct default file paths if not provided
    if odds_file is None:
        odds_file = f"DATA/historical_season_{season}.csv"
    
    if output_file is None:
        output_file = f"DATA/matched_season_{season}.csv"
    
    # Step 1: Scrape Baseball Reference schedule (golden source of truth)
    br_scraper = BaseballReferenceScraper()
    golden_schedule = br_scraper.scrape_season_schedule(season)
    
    if golden_schedule.empty:
        logger.error(f"Failed to scrape Baseball Reference schedule for {season}")
        return pd.DataFrame()
    
    logger.info(f"Golden schedule: {len(golden_schedule)} games")
    
    # Step 2: Load historical odds data
    try:
        odds_data = pd.read_csv(odds_file)
        logger.info(f"Loaded odds data: {len(odds_data)} games from {odds_file}")
    except FileNotFoundError:
        logger.warning(f"Odds file not found: {odds_file}")
        odds_data = pd.DataFrame(columns=['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML'])
    except Exception as e:
        logger.error(f"Error loading odds data: {e}")
        odds_data = pd.DataFrame(columns=['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML'])
    
    # Step 3: Prepare data for matching
    # Ensure both DataFrames have the same column structure for matching
    golden_schedule = golden_schedule.copy()
    golden_schedule['OU_Line'] = ''
    golden_schedule['Home_ML'] = ''
    golden_schedule['Away_ML'] = ''
    
    # Ensure odds data has all required columns
    if not odds_data.empty:
        required_columns = ['Date', 'Home', 'Away', 'Home_Score', 'Away_Score', 'OU_Line', 'Home_ML', 'Away_ML']
        for col in required_columns:
            if col not in odds_data.columns:
                odds_data[col] = ''
    
    # Step 4: Perform left join with chronological matching for doubleheaders
    if not odds_data.empty:
        # Sort both datasets by Date, Home, Away to ensure chronological order
        golden_schedule = golden_schedule.sort_values(['Date', 'Home', 'Away']).reset_index(drop=True)
        odds_data = odds_data.sort_values(['Date', 'Home', 'Away']).reset_index(drop=True)
        
        # Create a mapping from (Date, Home, Away) to list of odds entries
        odds_groups = odds_data.groupby(['Date', 'Home', 'Away']).apply(
            lambda x: x[['OU_Line', 'Home_ML', 'Away_ML']].to_dict('records')
        ).to_dict()
        
        # Apply the mapping chronologically (optimized version)
        game_counters = {}  # Track occurrence count for each game
        
        for idx, row in golden_schedule.iterrows():
            game_key = (row['Date'], row['Home'], row['Away'])
            
            if game_key in odds_groups:
                # Get the odds for this game (chronological order)
                odds_list = odds_groups[game_key]
                
                # Get current occurrence count for this game
                game_occurrence = game_counters.get(game_key, 0)
                
                # Use the corresponding odds entry (if available)
                if game_occurrence < len(odds_list):
                    odds_entry = odds_list[game_occurrence]
                    golden_schedule.at[idx, 'OU_Line'] = odds_entry['OU_Line']
                    golden_schedule.at[idx, 'Home_ML'] = odds_entry['Home_ML']
                    golden_schedule.at[idx, 'Away_ML'] = odds_entry['Away_ML']
                
                # Increment counter for next occurrence
                game_counters[game_key] = game_occurrence + 1
    

    
    # Step 5: Calculate matching statistics
    total_games = len(golden_schedule)
    matched_games = len(golden_schedule[golden_schedule['OU_Line'] != ''])
    match_rate = (matched_games / total_games * 100) if total_games > 0 else 0
    
    logger.info(f"Matching Statistics:")
    logger.info(f"  - Total games in golden schedule: {total_games}")
    logger.info(f"  - Games with odds data: {matched_games}")
    logger.info(f"  - Match rate: {match_rate:.1f}%")
    
    # Step 6: Save the matched data
    try:
        golden_schedule.to_csv(output_file, index=False)
        logger.info(f"✅ Saved matched data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving matched data: {e}")
    
    return golden_schedule

def match_all_historical_seasons(seasons: List[int] = None) -> None:
    """
    Match golden schedules with odds data for multiple seasons.
    
    Args:
        seasons: List of seasons to process (default: [2022, 2021, 2020, 2019, 2018, 2017, 2016])
    """
    if seasons is None:
        seasons = [2022, 2021, 2020, 2019, 2018, 2017, 2016]
    
    logger.info(f"=== Matching All Historical Seasons ===")
    logger.info(f"Target seasons: {seasons}")
    
    for season in seasons:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing season {season}")
        logger.info(f"{'='*50}")
        
        try:
            matched_data = match_golden_schedule_with_odds(season)
            
            if not matched_data.empty:
                logger.info(f"✅ Successfully processed season {season}")
            else:
                logger.warning(f"⚠️ No data processed for season {season}")
                
        except Exception as e:
            logger.error(f"❌ Error processing season {season}: {e}")
            continue
        
        # Add delay between seasons to be respectful
        if season != seasons[-1]:
            logger.info("⏳ Pausing 5 seconds between seasons...")
            time.sleep(5)
    
    logger.info(f"\n{'='*50}")
    logger.info("ALL SEASONS PROCESSING COMPLETE")
    logger.info(f"{'='*50}")
    
    # Print summary of generated files
    logger.info("\n📁 Generated matched files:")
    for season in seasons:
        output_file = f"DATA/matched_season_{season}.csv"
        logger.info(f"  - {output_file}")

def generate_date_list(start_date: str, end_date: str) -> list:
    """Generate a list of dates in YYYYMMDD format between start and end dates"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    return date_list

def aggregate_and_clean_output_files(input_pattern: str = "DATA/historical_season_*.csv", 
                                   output_file: str = "DATA/aggregated_historical_data.csv",
                                   filter_postponed: bool = True) -> None:
    """
    Aggregate multiple output files into a single file and optionally filter out postponed/cancelled games.
    
    Args:
        input_pattern: Glob pattern to match input files (default: "DATA/historical_season_*.csv")
        output_file: Path to the output aggregated file
        filter_postponed: If True, remove rows where both Home_Score and Away_Score are 0.0 (postponed/cancelled games)
    """
    import glob
    import pandas as pd
    from pathlib import Path
    
    logger.info(f"=== Aggregating and Cleaning Output Files ===")
    logger.info(f"Input pattern: {input_pattern}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Filter postponed games: {filter_postponed}")
    
    # Find all matching files
    matching_files = glob.glob(input_pattern)
    if not matching_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(matching_files)} files to aggregate:")
    for file in matching_files:
        logger.info(f"  - {file}")
    
    # Read and combine all files
    all_data = []
    total_rows_before = 0
    
    for file_path in matching_files:
        try:
            df = pd.read_csv(file_path)
            total_rows_before += len(df)
            all_data.append(df)
            logger.info(f"Read {len(df)} rows from {file_path}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
    
    if not all_data:
        logger.error("No data could be read from any files")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data: {len(combined_df)} total rows")
    
    # Remove duplicates if any
    initial_rows = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    if len(combined_df) < initial_rows:
        logger.info(f"Removed {initial_rows - len(combined_df)} duplicate rows")
    
    # Filter out postponed/cancelled games if requested
    if filter_postponed:
        initial_rows = len(combined_df)
        # Filter out rows where both Home_Score and Away_Score are 0.0
        postponed_mask = (combined_df['Home_Score'] == 0.0) & (combined_df['Away_Score'] == 0.0)
        postponed_count = postponed_mask.sum()
        
        if postponed_count > 0:
            combined_df = combined_df[~postponed_mask]
            logger.info(f"Filtered out {postponed_count} postponed/cancelled games (0-0 scores)")
            logger.info(f"Remaining games: {len(combined_df)}")
        else:
            logger.info("No postponed/cancelled games found to filter")
    
    # Sort by date for better organization
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.sort_values('Date', ascending=False)
    
    # Save the aggregated and cleaned data
    try:
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        combined_df.to_csv(output_file, index=False)
        logger.info(f"✅ Successfully saved aggregated data to {output_file}")
        
        # Print summary statistics
        logger.info(f"\n📊 Summary:")
        logger.info(f"  - Total files processed: {len(matching_files)}")
        logger.info(f"  - Total rows before aggregation: {total_rows_before}")
        logger.info(f"  - Total rows after aggregation: {len(combined_df)}")
        if filter_postponed:
            logger.info(f"  - Postponed/cancelled games removed: {postponed_count}")
        
        # Date range information
        if len(combined_df) > 0:
            earliest_date = combined_df['Date'].min().strftime('%Y-%m-%d')
            latest_date = combined_df['Date'].max().strftime('%Y-%m-%d')
            logger.info(f"  - Date range: {earliest_date} to {latest_date}")
        
        # Team statistics
        unique_teams = set(combined_df['Home'].unique()) | set(combined_df['Away'].unique())
        logger.info(f"  - Unique teams: {len(unique_teams)}")
        
    except Exception as e:
        logger.error(f"Error saving aggregated file {output_file}: {e}")
        return
    
    logger.info(f"=== Aggregation Complete ===")

def aggregate_matched_seasons(input_pattern: str = "DATA/matched_season_*.csv", 
                            output_file: str = "DATA/aggregated_matched_data.csv",
                            filter_postponed: bool = True) -> None:
    """
    Aggregate multiple matched season files into a single file and optionally filter out postponed/cancelled games.
    
    Args:
        input_pattern: Glob pattern to match input files (default: "DATA/matched_season_*.csv")
        output_file: Path to the output aggregated file
        filter_postponed: If True, remove rows where both Home_Score and Away_Score are empty or 0 (postponed/cancelled games)
    """
    import glob
    from pathlib import Path
    
    logger.info(f"=== Aggregating Matched Season Files ===")
    logger.info(f"Input pattern: {input_pattern}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Filter postponed games: {filter_postponed}")
    
    # Find all matching files
    matching_files = glob.glob(input_pattern)
    if not matching_files:
        logger.error(f"No files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(matching_files)} files to aggregate:")
    for file in matching_files:
        logger.info(f"  - {file}")
    
    # Read and combine all files
    all_data = []
    total_rows_before = 0
    
    for file_path in matching_files:
        try:
            df = pd.read_csv(file_path)
            total_rows_before += len(df)
            all_data.append(df)
            logger.info(f"Read {len(df)} rows from {file_path}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
    
    if not all_data:
        logger.error("No data could be read from any files")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data: {len(combined_df)} total rows")
    
    # Remove duplicates if any
    initial_rows = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    if len(combined_df) < initial_rows:
        logger.info(f"Removed {initial_rows - len(combined_df)} duplicate rows")
    
    # Filter out postponed/cancelled games if requested
    if filter_postponed:
        initial_rows = len(combined_df)
        # Filter out rows where both Home_Score and Away_Score are empty or 0
        postponed_mask = (
            (combined_df['Home_Score'].astype(str).str.strip() == '') & 
            (combined_df['Away_Score'].astype(str).str.strip() == '')
        ) | (
            (combined_df['Home_Score'] == 0) & (combined_df['Away_Score'] == 0)
        )
        postponed_count = postponed_mask.sum()
        
        if postponed_count > 0:
            combined_df = combined_df[~postponed_mask]
            logger.info(f"Filtered out {postponed_count} postponed/cancelled games")
            logger.info(f"Remaining games: {len(combined_df)}")
        else:
            logger.info("No postponed/cancelled games found to filter")
    
    # Sort by date for better organization
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df = combined_df.sort_values('Date', ascending=False)
    
    # Save the aggregated and cleaned data
    try:
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        combined_df.to_csv(output_file, index=False)
        logger.info(f"✅ Successfully saved aggregated matched data to {output_file}")
        
        # Print summary statistics
        logger.info(f"\n📊 Summary:")
        logger.info(f"  - Total files processed: {len(matching_files)}")
        logger.info(f"  - Total rows before aggregation: {total_rows_before}")
        logger.info(f"  - Total rows after aggregation: {len(combined_df)}")
        if filter_postponed:
            logger.info(f"  - Postponed/cancelled games removed: {postponed_count}")
        
        # Date range information
        if len(combined_df) > 0:
            earliest_date = combined_df['Date'].min().strftime('%Y-%m-%d')
            latest_date = combined_df['Date'].max().strftime('%Y-%m-%d')
            logger.info(f"  - Date range: {earliest_date} to {latest_date}")
        
        # Team statistics
        unique_teams = set(combined_df['Home'].unique()) | set(combined_df['Away'].unique())
        logger.info(f"  - Unique teams: {len(unique_teams)}")
        
        # Odds coverage statistics
        games_with_odds = len(combined_df[combined_df['OU_Line'] != ''])
        odds_coverage = (games_with_odds / len(combined_df) * 100) if len(combined_df) > 0 else 0
        logger.info(f"  - Games with odds data: {games_with_odds} ({odds_coverage:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error saving aggregated file {output_file}: {e}")
        return
    
    logger.info(f"=== Aggregation Complete ===")

def run_single_season_matching(season: int) -> None:
    """
    Run the matching process for a single season.
    
    Args:
        season: Year to process (e.g., 2021)
    """
    logger.info(f"=== Processing Single Season: {season} ===")
    
    # Check if odds file exists
    odds_file = f"DATA/historical_season_{season}.csv"
    if not os.path.exists(odds_file):
        logger.error(f"Odds file not found: {odds_file}")
        logger.info("Available odds files:")
        for file in os.listdir("DATA"):
            if file.startswith("historical_season_") and file.endswith(".csv"):
                logger.info(f"  - DATA/{file}")
        return
    
    try:
        # Run the matching process
        matched_data = match_golden_schedule_with_odds(season)
        
        if not matched_data.empty:
            logger.info(f"✅ Successfully processed season {season}")
            logger.info(f"  - Total games: {len(matched_data)}")
            games_with_odds = len(matched_data[matched_data['OU_Line'] != ''])
            odds_coverage = (games_with_odds / len(matched_data) * 100) if len(matched_data) > 0 else 0
            logger.info(f"  - Games with odds: {games_with_odds} ({odds_coverage:.1f}%)")
        else:
            logger.warning(f"⚠️ No data processed for season {season}")
            
    except Exception as e:
        logger.error(f"❌ Error processing season {season}: {e}")

def main():
    """Main function to match historical MLB odds data with golden schedules"""
    logger.info("=== Starting Historical MLB Data Matching ===")
    logger.info("Strategy: Match existing historical odds data with Baseball Reference golden schedules")
    logger.info("No scraping required - using existing historical season files")
    
    # Seasons to process (2018-2022) - we have odds data for these years
    seasons_to_process = [2022, 2021, 2020, 2019, 2018]
    
    logger.info(f"Target seasons: {seasons_to_process}")
    logger.info("Available historical odds files:")
    for season in seasons_to_process:
        odds_file = f"DATA/historical_season_{season}.csv"
        logger.info(f"  - {odds_file}")
    
    # Step 1: Match golden schedules with odds data
    logger.info(f"\n{'='*60}")
    logger.info("MATCHING GOLDEN SCHEDULES WITH ODDS DATA")
    logger.info(f"{'='*60}")
    
    try:
        match_all_historical_seasons(seasons_to_process)
        logger.info("✅ Successfully completed schedule matching")
    except Exception as e:
        logger.error(f"❌ Error during schedule matching: {e}")
    
    # Step 2: Aggregate all matched data
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATING MATCHED DATA")
    logger.info(f"{'='*60}")
    
    try:
        aggregate_matched_seasons(
            input_pattern="DATA/matched_season_*.csv",
            output_file="DATA/aggregated_matched_data.csv",
            filter_postponed=True
        )
        logger.info("✅ Successfully completed data aggregation")
    except Exception as e:
        logger.error(f"❌ Error during data aggregation: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info("HISTORICAL DATA MATCHING COMPLETE")
    logger.info(f"{'='*60}")
    
    logger.info("\n📁 Generated files:")
    logger.info("Matched data files:")
    for season in seasons_to_process:
        output_file = f"DATA/matched_season_{season}.csv"
        logger.info(f"  - {output_file}")
    
    logger.info("Aggregated files:")
    logger.info("  - DATA/aggregated_matched_data.csv")
    
    logger.info("\n🎯 Next steps:")
    logger.info("1. Review the generated CSV files")
    logger.info("2. Validate data quality and completeness")
    logger.info("3. Integrate with your existing dataset")
    logger.info("4. Use the aggregated_matched_data.csv for your analysis")

if __name__ == "__main__":
    main() 

# Example usage functions:
# 
# 1. Match golden schedule with odds for a single season:
# match_golden_schedule_with_odds(2021)
#
# 2. Run single season matching with validation:
# run_single_season_matching(2021)
#
# 3. Match all historical seasons (2018-2022):
# match_all_historical_seasons([2022, 2021, 2020, 2019, 2018])
#
# 4. Aggregate matched data:
# aggregate_matched_seasons()
#
# 5. Run the complete workflow:
# main()