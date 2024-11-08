from re import X
from tkinter import Y
import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import LeagueDashTeamStats, LeagueGameLog
from nba_api.stats.static import teams
import time
import requests
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings("ignore")


class NBAPredictor:
    def fetch_team_stats(self, season='2024-25'):
        """Fetch team statistics with retries for network issues."""
        try:
            season_format = season if '-' in season and len(season) == 7 else None
            if not season_format:
                raise ValueError("Season format should be 'YYYY-YY'.")
            
            time.sleep(2)  # Rate-limiting delay
            stats = LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Base',
                per_mode_detailed='PerGame',
                timeout=30
            ).get_data_frames()[0]

            stats = self.calculate_four_factors(stats)
            elo_ratings = self.fetch_elo_ratings()

            print("Stats columns:", stats.columns)
            print("Elo ratings columns:", elo_ratings.columns)

            if 'TEAM_ABBREVIATION' in stats.columns and 'TEAM_ABBREVIATION' in elo_ratings.columns:
                stats = pd.merge(stats, elo_ratings, on='TEAM_ABBREVIATION', how='left')
                print("Merge successful with TEAM_ABBREVIATION.")
            elif 'TEAM_NAME' in stats.columns and 'team1' in elo_ratings.columns:
                # Fallback if TEAM_ABBREVIATION is missing
                elo_ratings.rename(columns={'team1': 'TEAM_NAME'}, inplace=True)
                stats = pd.merge(stats, elo_ratings, on='TEAM_NAME', how='left')
                print("Merge successful with TEAM_NAME as fallback.")
            else:
                # If neither TEAM_ABBREVIATION nor TEAM_NAME is available
                print("Warning: TEAM_ABBREVIATION or TEAM_NAME not found for merging.")
                print("Stats columns:", stats.columns)
                print("Elo ratings columns:", elo_ratings.columns)

            return stats

        except requests.exceptions.Timeout as e:
            print(f"Timeout occurred: {e}")
        except ValueError as e:
            print(f"Value error: {e}")
        except Exception as e:
            print(f"Error fetching team stats: {e}")
        return pd.DataFrame()

    def calculate_four_factors(self, stats_df):
        """
        Calculate Dean Oliver's Four Factors:
        1. Shooting (eFG%)
        2. Turnovers (TOV%)
        3. Rebounding (OREB%)
        4. Free Throws (FT Rate)
        """
        try:
            # 1. Effective Field Goal Percentage (eFG%)
            stats_df['eFG%'] = (stats_df['FGM'] + 0.5 * stats_df['FG3M']) / stats_df['FGA']
    
            # 2. Turnover Percentage (TOV%)
            stats_df['TOV%'] = stats_df['TOV'] / (stats_df['FGA'] + 0.44 * stats_df['FTA'] + stats_df['TOV'])
    
            # 3. Offensive Rebounding Percentage (OREB%)
            stats_df['OREB%'] = stats_df['OREB'] / (stats_df['OREB'] + stats_df['DREB_OPP'])
    
            # 4. Free Throw Rate (FTRate)
            stats_df['FTRate'] = stats_df['FTM'] / stats_df['FGA']
    
            # Calculate Four Factors Score (weighted according to Oliver's research)
            stats_df['FourFactorsScore'] = (
                0.4 * stats_df['eFG%'] +
                0.25 * stats_df['TOV%'] * -1 +  # Negative because lower TO% is better
                0.2 * stats_df['OREB%'] +
                0.15 * stats_df['FTRate']
            )
    
            return stats_df
    
        except Exception as e:
            print(f"Error calculating Four Factors: {e}")
            return stats_df

    def fetch_elo_ratings(self):
        """
        Fetch and process ELO ratings from FiveThirtyEight with error handling.
        """
        try:
            url = 'https://projects.fivethirtyeight.com/nba-model/nba_elo.csv'
            elo_data = pd.read_csv(url)
        
            # Validate columns
            if 'date' not in elo_data.columns or 'team1' not in elo_data.columns:
                raise ValueError("ELO data missing required columns.")
        
            elo_data['date'] = pd.to_datetime(elo_data['date'])
            latest_elo = elo_data.groupby('team1').last().reset_index()
            elo_ratings = latest_elo[['team1', 'elo1_pre']].rename(
                columns={'team1': 'TEAM_ABBREVIATION', 'elo1_pre': 'ELO'}
            )
            return elo_ratings

        except requests.exceptions.RequestException as e:
            print(f"Error fetching ELO ratings: {e}")
        except ValueError as e:
            print(f"Value error in ELO data: {e}")
        return pd.DataFrame()

    def fetch_team_stats(self, season='2024-25'):
        """Fetch team statistics with Four Factors calculations"""
        try:
            time.sleep(2)
    
            # Fetch detailed stats
            stats = LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense='Base',
                per_mode_detailed='PerGame',
                timeout=30
            ).get_data_frames()[0]
    
            # Calculate Four Factors
            stats = self.calculate_four_factors(stats)
    
            # Fetch ELO ratings
            elo_ratings = self.fetch_elo_ratings()
    
            # Merge with ELO ratings
            if not elo_ratings.empty:
                stats = pd.merge(
                    stats,
                    elo_ratings,
                    on='TEAM_ABBREVIATION',
                    how='left'
                )
    
            return stats
    
        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return pd.DataFrame()

    def fetch_game_logs(self, season='2024-25'):
        """
        Fetches game logs for all teams for a given NBA season.
        """
        try:
            # Fetch league-wide team game logs
            game_log = LeagueGameLog(
                season=season,
                player_or_team_abbreviation='T'  # 'T' for team logs
            ).get_data_frames()[0]
    
            # Ensure GAME_DATE is datetime
            game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])
    
            # Determine Home or Away games
            game_log['HOME_GAME'] = game_log['MATCHUP'].apply(
                lambda x: 1 if 'vs.' in x else 0
            )
    
            # Map win/loss to binary
            game_log['WIN'] = game_log['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
            # Get opponent team abbreviation
            game_log['OPP_TEAM_ABBREVIATION'] = game_log['MATCHUP'].apply(
                lambda x: self.get_opponent_team_abbr(x)
            )
    
            # Sort by TEAM_ABBREVIATION and GAME_DATE
            game_log.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'], inplace=True)
    
            return game_log
    
        except Exception as e:
            print(f"Error fetching game logs: {e}")
            return pd.DataFrame()

    def get_opponent_team_abbr(self, matchup):
        """
        Extracts the opponent team abbreviation from the matchup string.
        """
        try:
            parts = matchup.split()
            if len(parts) >= 3:
                return parts[-1]
            else:
                return None
        except Exception as e:
            print(f"Error extracting opponent team abbreviation: {e}")
            return None

    def prepare_game_features(self, game_logs, team_stats):
        try:
            if 'TEAM_ABBREVIATION' not in team_stats.columns:
                raise KeyError("Required columns missing for merging.")

            features = game_logs.copy()

            # Merge with Four Factors and ELO from team_stats
            features = pd.merge(
                features,
                team_stats[['TEAM_ABBREVIATION', 'eFG%', 'TOV%', 'OREB%', 'FTRate', 'FourFactorsScore', 'ELO']],
                on=['TEAM_ABBREVIATION'],
                how='left'
            )
    
            # Merge opponent ELO
            opp_elo = team_stats[['TEAM_ABBREVIATION', 'ELO']].rename(
                columns={'TEAM_ABBREVIATION': 'OPP_TEAM_ABBREVIATION', 'ELO': 'OPP_ELO'}
            )
            features = pd.merge(
                features,
                opp_elo,
                on='OPP_TEAM_ABBREVIATION',
                how='left'
            )
    
            # Calculate ELO difference between teams
            features['ELO_DIFF'] = features['ELO'] - features['OPP_ELO']
    
            # Calculate rolling averages
            rolling_features = [
                'eFG%', 'TOV%', 'OREB%', 'FTRate', 'FourFactorsScore'
            ]
    
            for feature in rolling_features:
                features[f'{feature}_ROLLING_10'] = (
                    features.groupby('TEAM_ABBREVIATION')[feature]
                    .rolling(10, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
    
            # Select final feature set
            self.feature_columns = [
                'HOME_GAME',
                'eFG%_ROLLING_10',
                'TOV%_ROLLING_10',
                'OREB%_ROLLING_10',
                'FTRate_ROLLING_10',
                'FourFactorsScore_ROLLING_10',
                'ELO_DIFF'
            ]
    
            # Drop rows with missing values
            features.dropna(subset=self.feature_columns + ['WIN'], inplace=True)
    
            X = features[self.feature_columns]
            y = features['WIN']
    
            return X, y

        except KeyError as e:
            print(f"Key error in prepare_game_features: {e}")
        except Exception as e:
            print(f"Error preparing game features: {e}")
        return None, None

def train_model(self, X, y):
    try:
        X_scaled = self.scaler.fit_transform(X)

        # Check if the dataset has enough samples for TimeSeriesSplit
        if X_scaled.shape[0] < 10:  # Adjust threshold based on needs
            raise ValueError("Not enough samples for TimeSeriesSplit and RFECV.")
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42, eval_metric='logloss'),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=500, random_state=42)
        }

        # Train each model and perform feature selection
        for nbapredictor, model in models.items():
            print(f"Training {nbapredictor}...")
            rfecv = RFECV(estimator=model, step=1, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy', n_jobs=-1)
            rfecv.fit(X_scaled, y)
            
            selected_features = [f for f, s in zip(self.feature_columns, rfecv.support_) if s]
            print(f"{nbapredictor} Selected features: {selected_features}")

            X_selected = X_scaled[:, rfecv.support_]

            # Cross-validation...
            print(f"{nbapredictor} Model training complete.")
        return True

    except ValueError as e:
        print(f"ValueError in training model: {e}")
    except Exception as e:
        print(f"Error training model: {e}")
    return False

# Function to compile historical data from multiple seasons
def compile_historical_data(start_season='2014-15', end_season='2024-25'):
    predictor = NBAPredictor()
    seasons = []
    start_year = int(start_season.split('-')[0])
    end_year = int(end_season.split('-')[0])

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        print(f"Fetching data for season: {season}...")
        season_data = predictor.compile_nba_team_data(season)
        if not season_data.empty:
            print(f"Data for {season} compiled successfully.")
            seasons.append(season_data)
        else:
            print(f"No data found for season: {season}")
        time.sleep(2)
        
    if seasons:
        print("All seasons compiled.")
        historical_data = pd.concat(seasons, ignore_index=True)
        return historical_data
    else:
        print("No seasonal data compiled.")
        return pd.DataFrame()

# Function to compile all data within the class
def compile_nba_team_data(self, season='2024-25'):
    """
        Compiles team statistics, game logs, and advanced metrics into a single DataFrame.
    
        Args:
            season (str): NBA season in 'YYYY-YY' format, e.g., '2024-25'.
        
        Returns:
        pd.DataFrame: Compiled DataFrame ready for modeling.
    """
    # Fetch team statistics
    team_stats = self.fetch_team_stats(season)
    
    # Pause to respect API rate limits
    time.sleep(1)
    
    # Fetch game logs
    game_logs = self.fetch_game_logs(season)
    
    # Fetch ELO ratings (already included in fetch_team_stats)
    # elo_ratings = self.fetch_elo_ratings()
    
    # Merge game logs with team statistics on TEAM_ABBREVIATION
    combined_data = pd.merge(
        game_logs,
        team_stats,
        on=['TEAM_ABBREVIATION'],
        how='left',
        suffixes=('', '_team')
    )
    
    # The ELO ratings are already merged in team_stats, so no need to merge again
    
    # Select and rename relevant columns
    combined_data = combined_data[[
        'TEAM_ABBREVIATION', 'TEAM_NAME',
        'GAME_ID', 'GAME_DATE', 'MATCHUP', 'HOME_GAME', 'REST_DAYS',
        'WL', 'PTS', 'PTS_OPP', 'PLUS_MINUS',
        'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST',
        'NET_RATING', 'TOV_PCT', 'REB_PCT', 'FT_RATE',
        'WINS_LAST_10', 'LOSSES_LAST_10',
        'ELO_DIFF'  # Advanced metrics
    ]]

    if 'TEAM_ABBREVIATION' in game_logs.columns and 'TEAM_ABBREVIATION' in team_stats.columns:
        combined_data = pd.merge(game_logs, team_stats, on='TEAM_ABBREVIATION', how='left', suffixes=('', '_team'))
    else:
        print("Error: Missing necessary identifiers ('TEAM_ABBREVIATION') for merging.")
        return pd.DataFrame()
    
    # Convert percentages to decimal form if necessary
    percentage_columns = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'TOV_PCT', 'REB_PCT', 'FT_RATE']
    for col in percentage_columns:
        if combined_data[col].max() > 1:
            combined_data[col] = combined_data[col] / 100.0
    
    # Handle missing values if any
    combined_data.fillna(0, inplace=True)
    
    return combined_data

# Adding compile_nba_team_data as a method of NBAPredictor class
NBAPredictor.compile_nba_team_data = compile_nba_team_data

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
    
    # No need to encode 'HOME_GAME' since it's already binary (1 for Home, 0 for Away)
    
    # Select features and target
    features = [
        'HOME_GAME',
        'eFG%_ROLLING_10',
        'TOV%_ROLLING_10',
        'OREB%_ROLLING_10',
        'FTRate_ROLLING_10',
        'FourFactorsScore_ROLLING_10',
        'ELO_DIFF'
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

    # Split the data into training and testing sets using time-based split
    # For simplicity, we'll use the first 80% as training and last 20% as testing
    split_index = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    models = {}

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42)
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

    # XGBoost Classifier
    xgb_model = XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    report_xgb = classification_report(y_test, y_pred_xgb)
    models['XGBoost'] = {
        'model': xgb_model,
        'accuracy': accuracy_xgb,
        'report': report_xgb
    }

    # Neural Network Classifier
    nn_model = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=500, random_state=42)
    nn_model.fit(X_train, y_train)
    y_pred_nn = nn_model.predict(X_test)
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    report_nn = classification_report(y_test, y_pred_nn)
    models['Neural Network'] = {
        'model': nn_model,
        'accuracy': accuracy_nn,
        'report': report_nn
    }

    return models, scaler

# Function to visualize feature importances
from sklearn.inspection import permutation_importance

def plot_feature_importances(model, features):
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances = pd.Series(importances, index=features).sort_values(ascending=False)
        else:
            # Use permutation importance if feature_importances_ is not available
            result = permutation_importance(model, X, Y, n_repeats=10, random_state=42, n_jobs=-1)
            feature_importances = pd.Series(result.importances_mean, index=features).sort_values(ascending=False)
        
        # Plotting feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importances.index, feature_importances.values)
        plt.xlabel('Importance Score')
        plt.title('Feature Importances')
        plt.show()

    except Exception as e:
        print(f"Error plotting feature importances: {e}")

    def predict_upcoming_games(self, recent_games, team_stats):
        """
        Predicts outcomes for upcoming games based on recent data.
        Args:
            recent_games (pd.DataFrame): Recent games data.
            team_stats (pd.DataFrame): Current team stats including ELO, Four Factors, etc.
        """
        print("Preparing data for upcoming games predictions...")

        # Preparing the data based on last 10 games and Four Factors
        upcoming_games = recent_games.copy()

        # Merge with team stats
        upcoming_games = pd.merge(
            upcoming_games,
            team_stats[['TEAM_ABBREVIATION', 'eFG%', 'TOV%', 'OREB%', 'FTRate', 'FourFactorsScore', 'ELO']],
            on=['TEAM_ABBREVIATION'],
            how='left'
        )

        # Calculate the ELO difference
        opp_elo = team_stats[['TEAM_ABBREVIATION', 'ELO']].rename(
            columns={'TEAM_ABBREVIATION': 'OPP_TEAM_ABBREVIATION', 'ELO': 'OPP_ELO'}
        )
        upcoming_games = pd.merge(upcoming_games, opp_elo, on='OPP_TEAM_ABBREVIATION', how='left')
        upcoming_games['ELO_DIFF'] = upcoming_games['ELO'] - upcoming_games['OPP_ELO']

        # Calculating rolling averages based on last 10 games
        for factor in ['eFG%', 'TOV%', 'OREB%', 'FTRate', 'FourFactorsScore']:
            upcoming_games[f'{factor}_ROLLING_10'] = (
                upcoming_games.groupby('TEAM_ABBREVIATION')[factor]
                .rolling(10, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )

        # Select features for prediction
        X_upcoming = upcoming_games[
            ['HOME_GAME', 'eFG%_ROLLING_10', 'TOV%_ROLLING_10', 'OREB%_ROLLING_10', 'FTRate_ROLLING_10', 
             'FourFactorsScore_ROLLING_10', 'ELO_DIFF']
        ].dropna()

        # Scale the features using the existing scaler
        X_upcoming_scaled = self.scaler.transform(X_upcoming)

        # Predict outcomes
        predictions = self.model.predict(X_upcoming_scaled)
        upcoming_games['PREDICTION'] = predictions
        upcoming_games['PREDICTED_WIN'] = upcoming_games['PREDICTION'].apply(lambda x: 'Win' if x == 1 else 'Loss')

        print("Predictions for upcoming games:")
        print(upcoming_games[['TEAM_ABBREVIATION', 'OPP_TEAM_ABBREVIATION', 'PREDICTED_WIN']])

        return upcoming_games[['TEAM_ABBREVIATION', 'OPP_TEAM_ABBREVIATION', 'PREDICTED_WIN']]
    
# Main Execution Code
if __name__ == "__main__":
    predictor = NBAPredictor()

    # Compile historical data and train model as before
    historical_data = compile_historical_data(start_season='2014-15', end_season='2024-25')
    if historical_data.empty:
        print("No historical data compiled. Exiting.")
    else:
        print("Preprocessing and training model...")
        X, y = preprocess_data(historical_data)
        predictor.train_model(X, y)
        print("Model training complete.")

        # Fetch recent games and team stats
        print("Fetching current season data for predictions...")
        recent_games = predictor.fetch_game_logs(season='2024-25')
        team_stats = predictor.fetch_team_stats(season='2024-25')

        # Predict upcoming games
        predictions = predictor.predict_upcoming_games(recent_games, team_stats)
        print(predictions)

from datetime import datetime

class NBAPredictor:

    # Existing methods...

    def fetch_todays_games(self):
        """Fetch NBA games scheduled for today"""
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"Fetching games scheduled for today: {today}")
        try:
            game_log = LeagueGameLog(season="2024-25", player_or_team_abbreviation="T")
            games_df = game_log.get_data_frames()[0]
            games_today = games_df[games_df['GAME_DATE'] == today]
            return games_today[['TEAM_ABBREVIATION', 'MATCHUP']]
        except Exception as e:
            print(f"Error fetching today's games: {e}")
            return pd.DataFrame()

    def compare_teams(self, team1_stats, team2_stats):
        """Compare two teams' ELO and Four Factors stats"""
        print("Comparing teams based on ELO and Four Factors...")
        
        # ELO Comparison
        elo1, elo2 = team1_stats['ELO'], team2_stats['ELO']
        favored_team_elo = team1_stats['TEAM_ABBREVIATION'] if elo1 > elo2 else team2_stats['TEAM_ABBREVIATION']
        print(f"ELO Comparison: {team1_stats['TEAM_ABBREVIATION']} ({elo1}) vs {team2_stats['TEAM_ABBREVIATION']} ({elo2}) - "
              f"{favored_team_elo} is favored based on ELO")

        # Four Factors Comparison
        for factor in ['eFG%', 'TOV%', 'OREB%', 'FTRate', 'FourFactorsScore']:
            stat1, stat2 = team1_stats[factor], team2_stats[factor]
            if factor == 'TOV%':
                favored_team = team1_stats['TEAM_ABBREVIATION'] if stat1 < stat2 else team2_stats['TEAM_ABBREVIATION']
            else:
                favored_team = team1_stats['TEAM_ABBREVIATION'] if stat1 > stat2 else team2_stats['TEAM_ABBREVIATION']
            print(f"{factor} Comparison: {team1_stats['TEAM_ABBREVIATION']} ({stat1}) vs {team2_stats['TEAM_ABBREVIATION']} ({stat2}) - "
                  f"{favored_team} is favored based on {factor}")

    def predict_todays_games(self):
        """Predict outcomes for today's games based on ELO and Four Factors"""
        games_today = self.fetch_todays_games()
        team_stats = self.fetch_team_stats(season="2024-25")
        
        if games_today.empty or team_stats.empty:
            print("No games or team stats available for predictions.")
            return

        for _, game in games_today.iterrows():
            team1, team2 = game['TEAM_ABBREVIATION'], game['MATCHUP'].split()[-1]
            team1_stats = team_stats[team_stats['TEAM_ABBREVIATION'] == team1].squeeze()
            team2_stats = team_stats[team_stats['TEAM_ABBREVIATION'] == team2].squeeze()

            if team1_stats.empty or team2_stats.empty:
                print(f"Stats not available for teams {team1} vs {team2}. Skipping...")
                continue

            print(f"Predicting outcome for {team1} vs {team2}:")
            self.compare_teams(team1_stats, team2_stats)
            print("-" * 50)

# Example main execution
if __name__ == "__main__":
    predictor = NBAPredictor()

    print("Predicting today's NBA games based on ELO and Four Factors:")
    predictor.predict_todays_games()


class NBAPredictor:

    # Existing methods...

    def calculate_elo(self, team_elo, opp_elo, team_win, K=20):
        """Calculate updated ELO rating based on game outcome."""
        # Calculate expected win probabilities
        expected_team = 1 / (1 + 10 ** ((opp_elo - team_elo) / 400))
        
        # Update ELOs based on actual results
        elo_change = K * (team_win - expected_team)
        new_team_elo = team_elo + elo_change
        
        return new_team_elo

    def calculate_four_factors(self, stats_df):
        """Calculate Four Factors based on provided stats"""
        if 'FGM' in stats_df.columns and 'FGA' in stats_df.columns:
            stats_df['eFG%'] = (stats_df['FGM'] + 0.5 * stats_df['FG3M']) / stats_df['FGA']
        else:
            stats_df['eFG%'] = None
            print("Warning: Missing fields for eFG% calculation")

        if 'TOV' in stats_df.columns and 'FGA' in stats_df.columns and 'FTA' in stats_df.columns:
            stats_df['TOV%'] = stats_df['TOV'] / (stats_df['FGA'] + 0.44 * stats_df['FTA'] + stats_df['TOV'])
        else:
            stats_df['TOV%'] = None
            print("Warning: Missing fields for TOV% calculation")

        if 'OREB' in stats_df.columns and 'DREB_OPP' in stats_df.columns:
            stats_df['OREB%'] = stats_df['OREB'] / (stats_df['OREB'] + stats_df['DREB_OPP'])
        else:
            stats_df['OREB%'] = None
            print("Warning: Missing fields for OREB% calculation")

        if 'FTM' in stats_df.columns and 'FGA' in stats_df.columns:
            stats_df['FTRate'] = stats_df['FTM'] / stats_df['FGA']
        else:
            stats_df['FTRate'] = None
            print("Warning: Missing fields for FTRate calculation")

        # Calculate Four Factors Score
        stats_df['FourFactorsScore'] = (
            0.4 * stats_df['eFG%'] +
            0.25 * stats_df['TOV%'] * -1 +  # Lower TOV% is better
            0.2 * stats_df['OREB%'] +
            0.15 * stats_df['FTRate']
        )

        return stats_df

    def compare_teams_with_factors(self, team1_stats, team2_stats):
        """Compare two teams based on updated ELO and Four Factors"""
        print("Comparing teams on ELO and Four Factors")

        # ELO comparison
        team1_elo, team2_elo = team1_stats['ELO'], team2_stats['ELO']
        favored_team_elo = team1_stats['TEAM_ABBREVIATION'] if team1_elo > team2_elo else team2_stats['TEAM_ABBREVIATION']
        print(f"ELO Comparison: {team1_stats['TEAM_ABBREVIATION']} ({team1_elo}) vs {team2_stats['TEAM_ABBREVIATION']} ({team2_elo}) - "
              f"{favored_team_elo} is favored based on ELO")

        # Four Factors Comparison
        for factor in ['eFG%', 'TOV%', 'OREB%', 'FTRate', 'FourFactorsScore']:
            stat1, stat2 = team1_stats.get(factor), team2_stats.get(factor)
            if stat1 is None or stat2 is None:
                print(f"Warning: Missing data for {factor} comparison")
                continue

            if factor == 'TOV%':
                favored_team = team1_stats['TEAM_ABBREVIATION'] if stat1 < stat2 else team2_stats['TEAM_ABBREVIATION']
            else:
                favored_team = team1_stats['TEAM_ABBREVIATION'] if stat1 > stat2 else team2_stats['TEAM_ABBREVIATION']
            print(f"{factor} Comparison: {team1_stats['TEAM_ABBREVIATION']} ({stat1}) vs {team2_stats['TEAM_ABBREVIATION']} ({stat2}) - "
                  f"{favored_team} is favored based on {factor}")
        print("-" * 50)

