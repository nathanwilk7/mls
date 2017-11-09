import pandas as pd
import pdb

def get_matches(filepath):
    matches = pd.read_csv(filepath)
    return matches

def get_teams_matches(filepath, team):
    matches = pd.read_csv(filepath)
    team_matches = matches[(matches.home_team_name == team) | (matches.away_team_name == team)]
    return team_matches

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

    num_matches = home_team_matches_home.shape[0] + home_team_matches_away.shape[0]
    if num_matches > 0:
        avg_home_team_matches = sum_home_team_matches / num_matches
    else:
        avg_home_team_matches_home = sum_home_team_matches
        
    #away_team_matches = matches[matches.away_team_name == team]
    seasons_matches = team_matches[team_matches.season_id == season]
    past_matches = seasons_matches[seasons_matches.date < date]
    return past_matches
