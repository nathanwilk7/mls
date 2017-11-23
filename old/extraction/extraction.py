import pandas as pd
import pdb

def get_teams_season_past_matches(filepath, team, season, date):
    matches = pd.read_csv(filepath)
    home_team_matches = matches[matches.home_team_name == team]
    
    home_team_matches_home = home_team_matches[home_team_matches.home_team_name == team]
    home_team_matches_away = home_team_matches[home_team_matches.away_team_name == team]
    
    home_team_matches_home.drop('home_team_name')
    home_team_matches_home.drop('away_team_name')
    home_team_matches_away.drop('home_team_name')
    home_team_matches_away.drop('away_team_name')
    
    sum_home_team_matches_home = home_team_matches_home.sum()
    sum_home_team_matches_away = home_team_matches_away.sum()
    sum_home_team_matches = sum_home_team_matches_home + sum_home_team_matches_away

    home_num_matches = home_team_matches_home.shape[0] + home_team_matches_away.shape[0]
    if home_num_matches > 0:
        avg_home_team_matches = sum_home_team_matches / home_num_matches
    else:
        avg_home_team_matches = sum_home_team_matches
        
    away_team_matches = matches[matches.away_team_name == team]
    
    away_team_matches_home = away_team_matches[away_team_matches.home_team_name == team]
    away_team_matches_away = away_team_matches[away_team_matches.away_team_name == team]
    
    away_team_matches_home.drop('home_team_name')
    away_team_matches_home.drop('away_team_name')
    away_team_matches_away.drop('home_team_name')
    away_team_matches_away.drop('away_team_name')
    
    sum_away_team_matches_home = away_team_matches_home.sum()
    sum_away_team_matches_away = away_team_matches_away.sum()
    sum_away_team_matches = sum_away_team_matches_home + sum_away_team_matches_away

    away_num_matches = away_team_matches_home.shape[0] + away_team_matches_away.shape[0]
    if away_num_matches > 0:
        avg_away_team_matches = sum_away_team_matches / away_num_matches
    else:
        avg_away_team_matches = sum_away_team_matches
    
    avg_team_matches = avg_home_team_matches + avg_away_team_matches
    return avg_team_matches, avg_home_team_matches, avg_away_team_matches
#     seasons_matches = avg_team_matches[team_matches.season_id == season]
#     past_matches = seasons_matches[seasons_matches.date < date]
#     return past_matches
