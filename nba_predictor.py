from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time
from nba_api.stats.endpoints import boxscoretraditionalv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Step 1: Data Collection
def get_nba_game_data():
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable='2022-23')
    games = gamefinder.get_data_frames()[0]
    columns = ['TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_DATE', 'MATCHUP', 'WL', 
               'PTS', 'FG_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 'PLUS_MINUS']
    games = games[columns]
    games['WIN'] = games['WL'].apply(lambda x: 1 if x == 'W' else 0)
    return games

# Step 1a: Historical Game Data
def get_historical_game_data(start_season='2014-15', end_season='2024-25'):
    seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(int(start_season.split('-')[0]), int(end_season.split('-')[0]) + 1)]
    all_games = []

    for season in seasons:
        print(f"Fetching games for season {season}")
        gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        season_games = gamefinder.get_data_frames()[0]
        all_games.append(season_games)
        
        # Respectful delay to avoid potential rate limiting
        time.sleep(1)

    # Concatenate data from all seasons
    games_df = pd.concat(all_games, ignore_index=True)
    return games_df

# Fetch data for the past 10 seasons
games_df = get_historical_game_data()

# Step 1c: Player Data
def get_player_stats_for_games(games_df):
    player_stats = []
    
    for _, game in games_df.iterrows():
        game_id = game['GAME_ID']
        print(f"Fetching player stats for Game ID {game_id}")
        
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            boxscore_df = boxscore.get_data_frames()[0]
            
            # Aggregate player stats to team-level stats
            team_stats = boxscore_df.groupby('TEAM_ID').agg({
                'PTS': 'sum',
                'REB': 'sum',
                'AST': 'sum',
                'TOV': 'sum',
                'FGM': 'sum',
                'FGA': 'sum',
                'FTM': 'sum',
                'FTA': 'sum',
                'FG3M': 'sum',
                'FG3A': 'sum',
                'PLUS_MINUS': 'mean'
            }).reset_index()
            team_stats['GAME_ID'] = game_id
            player_stats.append(team_stats)
        
        except Exception as e:
            print(f"Error fetching data for game {game_id}: {e}")
        
        # Respectful delay
        time.sleep(1)
    
    # Combine all player stats into a single DataFrame
    player_stats_df = pd.concat(player_stats, ignore_index=True)
    return player_stats_df

# Fetch player stats for each game
player_stats_df = get_player_stats_for_games(games_df)

# Step 1d: Merging for dataset
def integrate_game_and_player_data(games_df, player_stats_df):
    # Merge player statistics with the game data
    merged_df = games_df.merge(player_stats_df, on=['GAME_ID', 'TEAM_ID'], how='inner', suffixes=('', '_player'))
    
    # Ensure we have one row per game with both teams' statistics
    game_data = merged_df.pivot(index='GAME_ID', columns='TEAM_ID')
    
    # Flatten multi-level columns and rename for clarity
    game_data.columns = ['_'.join(map(str, col)).strip() for col in game_data.columns.values]
    game_data.reset_index(inplace=True)
    
    return game_data

# Integrate game-level data with player statistics
final_game_data = integrate_game_and_player_data(games_df, player_stats_df)


# Step 2: Prepare the DataFrame
games = get_nba_game_data()
X = games[['PTS', 'FG_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 'PLUS_MINUS']]
y = games['WIN']

# Step 3: Set up a Pipeline with Imputation and Scaling
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('scaler', StandardScaler())                 # Scale features
])

# Apply the pipeline to transform the feature data
X_preprocessed = pipeline.fit_transform(X)

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Step 5: Model Training (Random Forest Example)
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Train Random Forest Model
rf_model = train_random_forest(X_train, y_train)

# Step 6: Evaluation Metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability for the positive class (Win)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Evaluate the Random Forest Model
print("Random Forest Performance:")
evaluate_model(rf_model, X_test, y_test)

# Step 7: Predicting Winners for Upcoming Games
def predict_upcoming_games(game_data_list, model, pipeline):
    """
    Predict the winners of upcoming games.
    game_data_list: List of dictionaries, each with keys 'PTS', 'FG_PCT', 'FT_PCT', 'REB', 'AST', 'TOV', 'PLUS_MINUS'.
    model: Trained model for prediction.
    pipeline: Pipeline object used during training for preprocessing.
    """
    # Convert list of game data dictionaries to DataFrame
    new_games_df = pd.DataFrame(game_data_list)
    new_games_preprocessed = pipeline.transform(new_games_df)
    
    predictions = model.predict(new_games_preprocessed)
    prediction_probabilities = model.predict_proba(new_games_preprocessed)[:, 1]  # Probability of Win

    results = []
    for i, pred in enumerate(predictions):
        result = {
            "Game": i + 1,
            "Prediction": "Win" if pred == 1 else "Loss",
            "Win Probability": prediction_probabilities[i]
        }
        results.append(result)
    return results

# Example usage for predicting upcoming games
upcoming_games = [
    {
        'PTS': 112,
        'FG_PCT': 0.47,
        'FT_PCT': 0.76,
        'REB': 43,
        'AST': 24,
        'TOV': 13,
        'PLUS_MINUS': 4
    },
    {
        'PTS': 108,
        'FG_PCT': 0.45,
        'FT_PCT': 0.74,
        'REB': 40,
        'AST': 22,
        'TOV': 15,
        'PLUS_MINUS': -3
    }
]

# Predict winners for upcoming games
predictions = predict_upcoming_games(upcoming_games, rf_model, pipeline)
print("Predictions for Upcoming Games:")
for prediction in predictions:
    print(prediction)

# Step 11: Integrate Game Data with Aggregated Player Data
def integrate_game_and_player_data(games_df, player_stats_df):
    # Merge player statistics with the game data
    merged_df = games_df.merge(player_stats_df, on=['GAME_ID', 'TEAM_ID'], how='inner', suffixes=('', '_player'))
    
    # Ensure we have one row per game with both teams' statistics
    game_data = merged_df.pivot(index='GAME_ID', columns='TEAM_ID')
    
    # Flatten multi-level columns and rename for clarity
    game_data.columns = ['_'.join(map(str, col)).strip() for col in game_data.columns.values]
    game_data.reset_index(inplace=True)
    
    return game_data

# Step 12: Train the Model with Historical Data
def train_model_with_historical_data():
    # Fetch data for the past 10 seasons
    games_df = get_historical_game_data()
    player_stats_df = get_player_stats_for_games(games_df)

    # Integrate game-level and player-level statistics
    final_game_data = integrate_game_and_player_data(games_df, player_stats_df)

    # Prepare features and target for training
    X = final_game_data.drop(columns=['WIN'])  # Replace 'WIN' with the actual target column name
    y = final_game_data['WIN']

    # Preprocess and split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate model
    y_pred = rf_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return rf_model

# Train the model with historical data
if __name__ == "__main__":
    print("Training model with historical data...")
    model = train_model_with_historical_data()