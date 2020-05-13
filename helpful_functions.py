#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:27:05 2020

@author: petertea
"""


# Helpful functions


def top_15_predictions(entire_test_data, prediction_probs):
    '''
    Provides 15 All - NBA predictions adhering to conventional rules
    Decision boundary is no longer 0.5. Instead it is the top 6 G's, 6 F's and 3 C's
    
    Input:
    mydict: dictionary w/ All_NBA_Pos; year; prediction probs; test set index
    
    Output: 
        Test set indices on the players predicted to be All-NBA (9 * 15 = 135 predictions in total)
    '''
    #import pandas as pd
    import numpy as np
    
    entire_test_dat = entire_test_data.copy()
    
    # --> Convert to dataframe
    entire_test_dat['prediction_prob'] = prediction_probs
    entire_test_dat['test_index'] = entire_test_dat.index
    
    
    # --> For each year and position group, rank them in terms of Pr(Selection)
    grouped_dat = entire_test_dat.groupby(['Year','All_NBA_Pos']).apply(lambda x: x.sort_values(["prediction_prob"], ascending = False)).reset_index(drop=True)
    
    
    # --> Keep only the top 6 for the guards and forwards
    guards_and_forwards = grouped_dat[(grouped_dat ['All_NBA_Pos'] == "Guard" ) | (grouped_dat ['All_NBA_Pos'] == "Forward") ]

    to_keep1 = guards_and_forwards.groupby(['Year','All_NBA_Pos']).head(6)

    
    # --> For centers we only want to keep the top 3 each year
    centers = grouped_dat[grouped_dat['All_NBA_Pos'] == "Center"]

    to_keep2 = centers.groupby(['Year']).head(3)
    
    
    
    #--> Combine into one DF
    predicted_players = to_keep1.append(to_keep2).sort_values(['Year', 'All_NBA_Pos'])
    
    # --> Keep index of predicted allnba on original dataset
    all_nba_index = predicted_players['test_index']
    
    predicted_players.drop(['test_index'], axis = 1, inplace =True)
    
    # Now also get a DF with the test set w/ the prediction columns added
    entire_test_dat['binary_prediction'] = np.where(entire_test_data.index.isin(all_nba_index), 1, 0)
    
    entire_test_dat.drop(['test_index'], axis = 1, inplace =True)

    
    return predicted_players, entire_test_dat




def all_nba_test_report(complete_test_set):
    '''
    Purpose: Report proportion of true 1st, 2nd and 3rd All-NBA players identified by model
    
    Arguments:
    test_index: y test set index (of original data frame)
    entire_data: dataframe used to create train-test datasets
    y_pred: test set predictions of fitted model
    
    '''

    
    # Calculate sum of predictions for each all-nba team
    # Note: we assume 'binary_response' label for true response
    temp = complete_test_set.groupby(['All_NBA_Flag']).agg({'binary_prediction': 'sum', 'binary_response': 'sum'})
    
    
    # Aesthetic changes
    temp.rename_axis(index={"All_NBA_Flag": "All NBA Team"}, inplace = True)
    temp.rename(columns={"Prediction": "Prediction Total",
                                 "binary_response": "True Total"}, inplace = True)
    
    
    return temp





# Which players could the model not identify?
def players_missed(entire_dat):
    '''
    Returns dataframe of all the All-NBA players a model was unable to identify
    
    Arguments:
    test_index: y test set index (of original data frame)
    entire_data: dataframe used to create train-test datasets
    y_pred: test set predictions of fitted model
    
    '''
    
    temp = entire_dat[(entire_dat['binary_response'] ==  int(1)) & (entire_dat['binary_prediction'] == int(0)) ]
    
    temp.sort_values(by=['All_NBA_Flag', 'Year','All_NBA_Pos'], inplace = True)
    
    return temp





# 2020 Predictions
    
def predict_2020(positions, player_names, binary_prediction, probability_predictions):
    '''
    Purpose: return clean dataset of predicted All-NBA Winners
    
    Arguments:
    positions: np array of player positions
    player_names: np array of player names
    binary_prediction: np array of binary predictions
    probability_predictions: np array of probability of selection
    
    '''
    import pandas as pd
    import numpy as np
    
    nba_2020_pred = pd.DataFrame({"Position": positions,
                         "Probability":  probability_predictions, 
                         "Selection": binary_prediction,
                         "Player": player_names})
    
    
    final_predictions = nba_2020_pred.groupby(['Position'])
    temp = final_predictions.apply(lambda x: x.sort_values(['Probability'],
                                                   ascending = False).head(8))
    
    
    # Only select 5 centers instead of 8
    return temp.loc[np.logical_not(temp.index.isin(temp.index[5:8]))]
    
    