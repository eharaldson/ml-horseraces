import pandas as pd 
import numpy as np 
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Split the data into train and validation sets by grouping into races first and then splitting
def split_train_validation_data(df: pd.DataFrame, validation_split: float):

    grouped = df.groupby(['DateTime', 'Track Name'])

    num_races = len(grouped)
    validation_index = round(num_races*(1-validation_split))

    train_races = []
    validation_races = []

    i = 0

    for (dt, track), group_df in grouped:

        if i < validation_index:
            train_races.append(group_df)
        else:
            validation_races.append(group_df)

        i += 1

    train_data = pd.concat(train_races)
    validation_data = pd.concat(validation_races)

    return train_data, validation_data

# Attach model results to the race data using an sklearn model
def get_sklearn_model_results_for_races(df_validationing: pd.DataFrame, feature_columns: list[str], model, target_column: str = 'model_prediction'):
    
    grouped = df_validationing.groupby(['DateTime', 'Track Name'])

    races = []

    for (dt, track), group_df in grouped:

        race = group_df
        race_results = model.predict_proba(race[feature_columns])[:,1]

        race_results_sum = sum(race_results)
        race_results_norm = [res/race_results_sum for res in race_results]

        race[target_column] = race_results_norm   

        races.append(race)

    return races

# Returns average log loss score over all races as an evaluation of model accuracy
def average_log_loss_score(race_results: list[pd.DataFrame], target_column: str):
    epsilon = 0.000001
    sum_log_loss = 0
    for race_result in race_results:
        winner_probability = race_result.loc[race_result['win'] == True].iloc[0][target_column]
        sum_log_loss += -np.log(winner_probability+epsilon)
    return sum_log_loss / len(race_results)

def betting_results_threshold_method(race_results: list[pd.DataFrame], implied_prob_column: str, threshold: float, bet_size: float = 10):

    portfolio = [0]

    winning_races = []
    winnings = []

    losing_races = []
    losings = []

    for race_result in race_results:
        race_result['bet_on'] = (race_result['model_prediction'] / race_result[implied_prob_column]) > (threshold+1)
        horses_bet_on = race_result.loc[race_result['bet_on']]
        horse_winnings = horses_bet_on.apply(lambda x: -1*bet_size if x['win'] != True else bet_size*((1/x[implied_prob_column])-1), axis=1)
        total_winnings = horse_winnings.sum()

        portfolio.append(portfolio[-1] + total_winnings)

        if total_winnings > 0:
            winning_races.append(race_result)
            winnings.append(total_winnings)
        else:
            losing_races.append(race_result)
            losings.append(total_winnings)

    return portfolio, winning_races, winnings, losing_races, losings

def betting_results_top_bsp_method(race_results: list[pd.DataFrame], bet_returns_column: str, implied_prob_column: str, bet_size: float = 10):

    portfolio = [0]

    winning_races = []
    winnings = []

    losing_races = []
    losings = []

    for race_result in race_results:
        betfair_favourite = race_result.sort_values(by=implied_prob_column).iloc[-1]

        if betfair_favourite['win']:
            total_winnings = bet_size*((1/betfair_favourite[bet_returns_column]) - 1)
            winning_races.append(race_result)
            winnings.append(total_winnings)
        else:
            total_winnings = -bet_size
            losing_races.append(race_result)
            losings.append(total_winnings)

        portfolio.append(portfolio[-1] + total_winnings)

    return portfolio, winning_races, winnings, losing_races, losings