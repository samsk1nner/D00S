# Import necessary libraries
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import LeagueDashTeamStats, LeagueGameLog
from nba_api.stats.static import teams
import time  # To respect API rate limits
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Function to determine if a game is Home or Away
def determine_home_away(matchup):
    if 'vs.' in matchup:
        return 'Home'
    elif '@' in matchup:
        return 'Away'
    else:
        return 'Neutral'

# Function to fetch and compile team statistics
from nba_api.stats.endpoints import LeagueDashTeamStats
import inspect

# Function to fetch and compile team statistics
def fetch_team_statistics(season='2024-25'):
    """
    Fetches base and advanced team statistics for a given NBA season.
    
    Args:
        season (str): NBA season in 'YYYY-YY' format, e.g., '2023-24'.
        
    Returns:
        pd.DataFrame: Merged DataFrame containing relevant team statistics.
    """
    try:
        # Inspect available parameters for LeagueDashTeamStats
        params = inspect.signature(LeagueDashTeamStats.__init__).parameters
        print(f"Available parameters for LeagueDashTeamStats: {list(params.keys())}")
        
        # Fetch base team stats
        team_stats_base = LeagueDashTeamStats(
            season=season,
            measure_type='Base',
            per_mode_detailed='PerGame'
        ).get_data_frames()[0]
        
        # Pause to respect API rate limits
        time.sleep(1)
        
        # Fetch advanced team stats
        team_stats_advanced = LeagueDashTeamStats(
            season=season,
            measure_type='Advanced',
            per_mode_detailed='PerGame'
        ).get_data_frames()[0]
        
        # Merge base and advanced stats on TEAM_ID
        team_stats = pd.merge(
            team_stats_base,
            team_stats_advanced,
            on='TEAM_ID',
            suffixes=('_base', '_adv')
        )
        
        # Select and rename relevant columns
        team_stats = team_stats[[
            'TEAM_ID', 'TEAM_NAME_base',
            'FG_PCT_base', 'FG3_PCT_base', 'FT_PCT_base',
            'PTS_base', 'REB_base', 'AST_base',
            'NET_RATING_adv', 'TOV_PCT_adv', 'REB_PCT_adv', 'FT_RATE_adv'
        ]]
        
        team_stats.rename(columns={
            'TEAM_NAME_base': 'TEAM_NAME',
            'FG_PCT_base': 'FG_PCT',
            'FG3_PCT_base': 'FG3_PCT',
            'FT_PCT_base': 'FT_PCT',
            'PTS_base': 'PTS',
            'REB_base': 'REB',
            'AST_base': 'AST',
            'NET_RATING_adv': 'NET_RATING',
            'TOV_PCT_adv': 'TOV_PCT',
            'REB_PCT_adv': 'REB_PCT',
            'FT_RATE_adv': 'FT_RATE'
        }, inplace=True)
        
        return team_stats
    
    except TypeError as te:
        print(f"TypeError: {te}")
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Error fetching team statistics: {e}")
        return pd.DataFrame()

# Function to fetch and compile game logs
def fetch_game_logs(season='2024-25'):
    """
    Fetches game logs for all teams for a given NBA season.
    
    Args:
        season (str): NBA season in 'YYYY-YY' format, e.g., '2024-25'.
        
    Returns:
        pd.DataFrame: DataFrame containing game logs with additional features.
    """
    try:
        # Fetch league-wide team game logs
        league_game_log = LeagueGameLog(
            season=season,
            player_or_team_abbreviation='T'  # 'T' for team logs
        ).get_data_frames()[0]
        
        # Ensure GAME_DATE is datetime
        league_game_log['GAME_DATE'] = pd.to_datetime(league_game_log['GAME_DATE'])
        
        # Determine Home or Away games
        league_game_log['HOME_AWAY'] = league_game_log['MATCHUP'].apply(determine_home_away)
        
        # Calculate Plus/Minus if not already present
        if 'PLUS_MINUS' not in league_game_log.columns:
            league_game_log['PLUS_MINUS'] = league_game_log['PTS'] - league_game_log['PTS_OPP']
        
        # Sort by TEAM_ID and GAME_DATE
        league_game_log.sort_values(['TEAM_ID', 'GAME_DATE'], inplace=True)
        
        # Calculate Rest Days between games
        league_game_log['PREV_GAME_DATE'] = league_game_log.groupby('TEAM_ID')['GAME_DATE'].shift(1)
        league_game_log['REST_DAYS'] = (league_game_log['GAME_DATE'] - league_game_log['PREV_GAME_DATE']).dt.days.fillna(0)
        
        # Calculate Win/Loss in the Last 10 Games
        league_game_log['WIN'] = league_game_log['WL'].apply(lambda x: 1 if x == 'W' else 0)
        league_game_log['LOSS'] = league_game_log['WL'].apply(lambda x: 1 if x == 'L' else 0)
        league_game_log['WINS_LAST_10'] = league_game_log.groupby('TEAM_ID')['WIN'].rolling(window=10, min_periods=1).sum().reset_index(0, drop=True)
        league_game_log['LOSSES_LAST_10'] = league_game_log.groupby('TEAM_ID')['LOSS'].rolling(window=10, min_periods=1).sum().reset_index(0, drop=True)
        
        return league_game_log
    
    except Exception as e:
        print(f"Error fetching game logs: {e}")
        return pd.DataFrame()

# Function to fetch ELO ratings
def fetch_elo_ratings():
    """
    Fetches NBA ELO ratings from FiveThirtyEight.
    
    Returns:
        pd.DataFrame: DataFrame containing ELO ratings.
    """
    try:
        elo_url = 'https://projects.fivethirtyeight.com/nba-model/nba_elo.csv'
        elo_data = pd.read_csv(elo_url)
        elo_data['date'] = pd.to_datetime(elo_data['date'])
        return elo_data
    except Exception as e:
        print(f"Error fetching ELO ratings: {e}")
        return pd.DataFrame()

# Function to fetch RAPTOR ratings (hypothetical implementation)
#def fetch_raptor_ratings():
    """
    Fetches NBA RAPTOR ratings.
    
    Returns:
        pd.DataFrame: DataFrame containing RAPTOR ratings.
    """
#    try:
        # Hypothetical URL; replace with actual data source
#        raptor_url = 'https://example.com/nba_raptor.csv'
#        raptor_data = pd.read_csv(raptor_url)
#        raptor_data['date'] = pd.to_datetime(raptor_data['date'])
#        return raptor_data
#    except Exception as e:
#        print(f"Error fetching RAPTOR ratings: {e}")
#        return pd.DataFrame()

# Function to compile all data
def compile_nba_team_data(season='2024-25'):
    """
        Compiles team statistics, game logs, and advanced metrics into a single DataFrame.
    
        Args:
            season (str): NBA season in 'YYYY-YY' format, e.g., '2024-25'.
        
        Returns:
        pd.DataFrame: Compiled DataFrame ready for modeling.
    """
    # Fetch team statistics
    team_stats = fetch_team_statistics(season)
    
    # Pause to respect API rate limits
    time.sleep(1)
    
    # Fetch game logs
    game_logs = fetch_game_logs(season)
    
    # Fetch ELO ratings
    elo_ratings = fetch_elo_ratings()
    
    # Fetch RAPTOR ratings (if available)
    raptor_ratings = fetch_raptor_ratings()
    
    # Merge game logs with team statistics on TEAM_ID
    combined_data = pd.merge(
        game_logs,
        team_stats,
        on='TEAM_ID',
        how='left',
        suffixes=('', '_team')
    )
    
    # Merge ELO ratings
    # Assuming 'team1' is the home team and 'team2' is the away team
    # For simplicity, we'll merge ELO_DIFF based on home team
    # Adjust accordingly based on ELO data structure
    combined_data = pd.merge(
        combined_data,
        elo_ratings[['team1', 'team2', 'elo1_pre', 'elo2_pre']],
        left_on='TEAM_NAME',
        right_on='team1',
        how='left'
    )
    
    # Calculate ELO difference (Home team ELO - Away team ELO)
    combined_data['ELO_DIFF'] = combined_data['elo1_pre'] - combined_data['elo2_pre']
    
    # Drop unnecessary ELO columns
    combined_data.drop(['team1', 'team2', 'elo1_pre', 'elo2_pre'], axis=1, inplace=True)
    
    # If RAPTOR data is available, merge it similarly
    if not raptor_ratings.empty:
        combined_data = pd.merge(
            combined_data,
            raptor_ratings[['team', 'raptor']],
            left_on='TEAM_NAME',
            right_on='team',
            how='left'
        )
        # Drop duplicate team column
        combined_data.drop(['team'], axis=1, inplace=True)
    else:
        combined_data['RAPTOR'] = 0  # Placeholder if RAPTOR not available
    
    # Select and rename relevant columns
    combined_data = combined_data[[
        'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
        'GAME_ID', 'GAME_DATE', 'MATCHUP', 'HOME_AWAY', 'REST_DAYS',
        'WL', 'PTS', 'PTS_OPP', 'PLUS_MINUS',
        'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST',
        'NET_RATING', 'TOV_PCT', 'REB_PCT', 'FT_RATE',
        'WINS_LAST_10', 'LOSSES_LAST_10',
        'ELO_DIFF', 'RAPTOR'  # Advanced metrics
    ]]
    
    # Convert percentages to decimal form if necessary
    percentage_columns = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'TOV_PCT', 'REB_PCT', 'FT_RATE']
    for col in percentage_columns:
        if combined_data[col].max() > 1:
            combined_data[col] = combined_data[col] / 100.0
    
    # Handle missing values if any
    combined_data.fillna(0, inplace=True)
    
    return combined_data

# Function to compile historical data from multiple seasons
def compile_historical_data(start_season='2014-15', end_season='2024-25'):
    """
    Compiles historical data from start_season to end_season.
    
    Args:
        start_season (str): Starting NBA season in 'YYYY-YY' format.
        end_season (str): Ending NBA season in 'YYYY-YY' format.
        
    Returns:
        pd.DataFrame: Compiled historical data.
    """
    seasons = []
    start_year = int(start_season.split('-')[0])
    end_year = int(end_season.split('-')[0])
    
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        print(f"Fetching data for season: {season}")
        season_data = compile_nba_team_data(season)
        if not season_data.empty:
            seasons.append(season_data)
        else:
            print(f"No data found for season: {season}")
        # Pause between seasons to respect rate limits
        time.sleep(2)
    
    historical_data = pd.concat(seasons, ignore_index=True)
    return historical_data

# Function to preprocess and prepare data for modeling
def preprocess_data(historical_data):
    """
    Preprocesses the historical data for machine learning modeling.
    
    Args:
        historical_data (pd.DataFrame): Compiled historical NBA data.
        
    Returns:
        pd.DataFrame, pd.Series: Feature matrix X and target vector y.
    """
    # Convert 'WL' to binary target variable
    historical_data['WIN'] = historical_data['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # Encode 'HOME_AWAY' as binary variables
    historical_data = pd.get_dummies(historical_data, columns=['HOME_AWAY'], drop_first=True)
    
    # Select features and target
    features = [
        'REST_DAYS', 'PTS', 'PTS_OPP', 'PLUS_MINUS',
        'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST',
        'NET_RATING', 'TOV_PCT', 'REB_PCT', 'FT_RATE',
        'WINS_LAST_10', 'LOSSES_LAST_10',
        'ELO_DIFF', 'RAPTOR',
        'HOME_AWAY_Home', 'HOME_AWAY_Neutral'
    ]
    X = historical_data[features]
    y = historical_data['WIN']
    
    return X, y

# Function to train and evaluate models
def train_and_evaluate(X, y):
    """
    Trains and evaluates machine learning models.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        
    Returns:
        dict: Trained models with their evaluation metrics.
    """
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the feature matrix
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame for easier handling
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    models = {}
    
    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf)
    models['Random Forest'] = {
        'model': rf_model,
        'accuracy': accuracy_rf,
        'report': report_rf
    }
    
    # Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    report_gb = classification_report(y_test, y_pred_gb)
    models['Gradient Boosting'] = {
        'model': gb_model,
        'accuracy': accuracy_gb,
        'report': report_gb
    }
    
    return models, scaler

# Function to visualize feature importances
def plot_feature_importances(model, features):
    """
    Plots feature importances for a given model.
    
    Args:
        model: Trained machine learning model.
        features (list): List of feature names.
    """
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=features).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title('Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Compile historical data
    historical_data = compile_historical_data(start_season='2014-15', end_season='2024-25')
    print("Historical data compilation complete.")
    
    # Save historical data
    historical_data.to_csv('nba_historical_team_data_2014-15_to_2024-25.csv', index=False)
    print("Historical data saved to 'nba_historical_team_data_2014-15_to_2024-25.csv'")
    
    # Preprocess data
    X, y = preprocess_data(historical_data)
    print("Data preprocessing complete.")
    
    # Train and evaluate models
    models, scaler = train_and_evaluate(X, y)
    
    # Display model accuracies
    for model_name, details in models.items():
        print(f"{model_name} Accuracy: {details['accuracy'] * 100:.2f}%")
        print(details['report'])
    
    # Plot feature importances for Random Forest
    plot_feature_importances(models['Random Forest']['model'], X.columns)
    
    # Save the best model and scaler
    best_model = models['Random Forest']['model']  # Assuming Random Forest performed better
    joblib.dump(best_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Trained model and scaler saved.")