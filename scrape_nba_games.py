print("Hello Noah's lock for winning in sports betting.")
import datetime
import re
import json
import schedule
import time
from sqlalchemy import create_engine

def scrape_nba_games():
    url = "https://www.nba.com/stats/scoreboard/"
    headers = {
        "User-Agent": "YourAppName/1.0 (contact@example.com)"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    pattern = re.compile(r"window\.__INITIAL_STATE__=({.*});")
    script = soup.find("script", text=pattern)
    if not script:
        print("Failed to find the data script.")
        return

    match = pattern.search(script.string)
    if not match:
        print("Failed to parse the data script.")
        return

    data_json = match.group(1)
    data = json.loads(data_json)

    try:
        games = data['scoreboard']['games']
    except KeyError:
        print("Failed to locate game data in JSON.")
        return

    game_data = []
    for game in games:
        game_info = {
            "GameID": game.get("gameId"),
            "StartTime": game.get("startTimeEastern"),
            "Status": game.get("gameStatusText"),
            "HomeTeam": game.get("homeTeam", {}).get("teamName"),
            "HomeScore": game.get("homeTeam", {}).get("score"),
            "AwayTeam": game.get("awayTeam", {}).get("teamName"),
            "AwayScore": game.get("awayTeam", {}).get("score"),
        }
        game_data.append(game_info)

    df = pd.DataFrame(game_data)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Save to CSV
    df.to_csv(f"nba_games_{current_date}.csv", index=False)

    # Save to SQLite
    engine = create_engine('sqlite:///nba_games.db')
    df.to_sql('games', con=engine, if_exists='append', index=False)

    print(f"Scraped data for {len(df)} games and saved to 'nba_games_{current_date}.csv' and 'nba_games.db'")

# Schedule the scraping every hour
schedule.every(1).hours.do(scrape_nba_games)

print("Starting NBA data scraper...")
scrape_nba_games()  # Initial scrape

while True:
    schedule.run_pending()
    time.sleep(1)