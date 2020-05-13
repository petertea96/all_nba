#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:19:58 2020

@author: petertea
"""
##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Purpose: All - NBA predictions
# Testing 5 Classification algorithms on predicting All-NBA players
##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####

# Set working directory
from os import chdir, getcwd
wd = '/Users/petertea/Documents/Sports-Analytics/NBA/per_100_poss/Project Scripts/'

chdir(wd)
getcwd()

# Import Libraries
import numpy as np
import pandas as pd

# --> Data Visualization library
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

pd.set_option('display.max_columns', None)
#plt.rcParams['figure.figsize'] = (20.0, 10.0)


##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Step I. Read Data Set and pre-process
##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
per_100_dat = pd.read_csv('./Data/Clean_All_NBA_Data.csv')

# Lets consider players who played a threshold number of games and minutes/game
threshold_games = 10
threshold_mins = 25

merged_dat = per_100_dat[ ( (per_100_dat["G"]) > threshold_games ) & ( (per_100_dat["MP"]/per_100_dat["G"]) > threshold_mins )]

merged_dat.reset_index(drop=True, inplace = True)


# --> Sanity Check
# --> Check that there are 15 All NBA selections each year
merged_dat[merged_dat['All_NBA_Flag'] != 'Not Selected'].groupby(['Year'])['Year'].count()




# We don't want to mess around with aliases, so lets work on a copy instead
allnba_dat = merged_dat.copy()

# Drop uninteresting columns
allnba_dat = allnba_dat.drop(["Season", "PlayerId", "Pos", "Tm", "key_id", "All_NBA_Team", "Team30"], axis = 1)

# Reset row index
allnba_dat.reset_index(drop=True, inplace = True)

# Add in All-NBA binary response
allnba_dat['binary_response'] = np.where(allnba_dat['All_NBA_Flag'] == "Not Selected", 0, 1)


# Final sanity check that we didn't lose any rows 
allnba_dat.shape[0] == merged_dat.shape[0]






##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Step 2. # --> Train - Test split
##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####

# Choose predictors and response
X = allnba_dat.drop(['binary_response','Year', 'All_NBA_Flag', 'Player'],axis=1)

y_binary = allnba_dat['binary_response']

# Convert variable All_NBA_Pos into an integer variable, rather than string
from sklearn import preprocessing

position_encoder = preprocessing.LabelEncoder()
X['All_NBA_Pos'] = position_encoder.fit_transform(X['All_NBA_Pos'])




# --> Array of possible years
years = np.arange(start = 1989, stop = 2020)

import random
random.seed(824)


# --> randomly select 9 years from array (without replacement)
test_years = np.sort(random.sample(set(years), k = 9))

print("Test Years: {0}".format(test_years))

# --> Assign train years all elements of years not in test years
train_years = np.setdiff1d(years, test_years)
print("\nTrain Years: {0}".format(train_years))


# --> Get all row indices in train and test years
train_index = allnba_dat[allnba_dat['Year'].isin(train_years)].index
test_index = allnba_dat[allnba_dat['Year'].isin(test_years)].index


# --> Train - test split using these indices
X_train = X.iloc[train_index]
y_train = y_binary.iloc[train_index]

X_test = X.iloc[test_index]
y_test = y_binary.iloc[test_index]


# --> Check dimensions
X_train.shape
len(y_train)

X_test.shape
len(y_test)


# --> Sanity Check:
# See for yourself
check_train_dat = allnba_dat.iloc[np.array(X_train.index)]

train_prop = check_train_dat.groupby(['All_NBA_Flag']).count()['Player'] / check_train_dat.shape[0]


check_test_dat = allnba_dat.iloc[np.array(X_test.index)]

test_prop = check_test_dat.groupby(['All_NBA_Flag']).count()['Player'] / check_test_dat.shape[0]


print('Rows in Training data:{0}'.format(check_train_dat.shape[0]))
print(train_prop)
print()
print('Rows in Testing data:{0}'.format(check_test_dat.shape[0]))
print(test_prop)





##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Step 3. Fit models
##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####


##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Model 1.1 - Logistic Regression


# --> Fit model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# --> Evaluate performance
# Evaluate performance:
logreg_pred = logreg.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
logreg_confusion_matrix = confusion_matrix(y_test, logreg_pred)

print("Logistic Regression confusion matrix:")
print(logreg_confusion_matrix)

print("Logistic Regression precision report:")
print(classification_report(y_test,  logreg_pred))


logreg_pred_prob = logreg.predict_proba(X_test)[:,1]




### --> Real way to assess performance
entire_test_data = allnba_dat.iloc[test_index]


from helpful_functions import top_15_predictions, all_nba_test_report, players_missed


logreg_preds, complete_logreg_dat = top_15_predictions(entire_test_data, logreg_pred_prob)

logreg_performance = all_nba_test_report(complete_logreg_dat)


players_missed(complete_logreg_dat)



##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Step 3. Obtain 2020 Predictions
##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####

# Get scores for current 2020 season
current_dat = pd.read_csv('./Data/Clean_current_2020_data.csv')

# Lets consider players who played a threshold number of games and minutes/game
threshold = 10
current_dat = current_dat[ ( (current_dat["G"]) > threshold ) & ( (current_dat["MP"]/current_dat["G"]) > 25 )]


# Which features were used to fit model?
features_to_keep = X.columns.astype(str)

#Extract these features
features_2020 = current_dat[features_to_keep]
features_2020['All_NBA_Pos'] = position_encoder.fit_transform(features_2020['All_NBA_Pos'])
#features_2020.head(3)





# Make our predictions
logreg_predict_probs_2020 = logreg.predict_proba(features_2020)
logreg_predict_binary_2020 = logreg.predict(features_2020)


from helpful_functions import predict_2020

logreg_2020_predictions = predict_2020(positions=position_encoder.inverse_transform(features_2020['All_NBA_Pos']),
            player_names=current_dat['Player'],
            binary_prediction=logreg_predict_binary_2020,
            probability_predictions= logreg_predict_probs_2020[:,1])


logreg_2020_predictions.to_csv("log_reg_predictions.csv")

current_dat[(current_dat['Player'] == "Pascal Siakam") | (current_dat['Player'] == "Bam Adebayo")]
current_dat[current_dat['All_NBA_Pos'] == "Forward"].sort_values(by = ['PTS', 'MP'], ascending = False).head(20)






##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Model 1.2 - Random Forests and Decision Trees
# 

# Train single classification tree
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion="gini")

dtree.fit(X_train,y_train)

# Evaluate Decision Tree
predictions_dtree = dtree.predict(X_test)
print(classification_report(y_test,predictions_dtree))


# Fit Random Forest (an ensemble of Decision Trees)

#--> Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Train the model using the training sets 
rfc_binary = RandomForestClassifier(n_estimators=200, criterion = "gini")

rfc_binary.fit(X_train, y_train)

# Plot Feature Importance
feature_importance_binary = pd.DataFrame({'Importance': np.array(rfc_binary.feature_importances_),
                                      'Feature': np.array(X_train.columns) })

feature_importance_binary = feature_importance_binary.sort_values(by = ['Importance'], ascending=True).tail(30)


# Set size of figure:
plt.figure(figsize=(5,8))

# Change labels of feature names in dataset:

feature_importance_binary['Feature']=np.where(feature_importance_binary['Feature'] == 'ORtg', 'OFF Rating', 
         np.where(feature_importance_binary['Feature'] == 'DRtg', 'DEF Rating',
                  np.where(feature_importance_binary['Feature'] == 'Win_Prop', 'Team Performance',
                           np.where(feature_importance_binary['Feature'] =="All_NBA_Pos", "Position",
                  feature_importance_binary['Feature']))))



plt.barh(width = feature_importance_binary['Importance'],
       y = feature_importance_binary['Feature'], color = '#71004B', alpha = 0.75)


plt.xlabel('Feature Importance Score', fontweight = "bold")
plt.ylabel('Features',fontweight = "bold")
plt.title("Feature Importance for\n All-NBA Selection",
          fontsize = 16, fontweight = "bold")

plt.show()

#plt.savefig('feature_importance.png', bbox_inches='tight')


# Model Evaluation
rfc_pred_prob = rfc_binary.predict_proba(X_test)[:,1]


rfc_preds, complete_rfc_dat = top_15_predictions(entire_test_data, rfc_pred_prob )


rfc_performance = all_nba_test_report(complete_rfc_dat)


players_missed(complete_rfc_dat)



# --> Predict 2020
rfc_probs_2020 = rfc_binary.predict_proba(features_2020)
rfc__2020 = rfc_binary.predict(features_2020)


rfc_predictions_2020 = predict_2020(positions=position_encoder.inverse_transform(features_2020['All_NBA_Pos']),
            player_names=current_dat['Player'],
            binary_prediction=rfc__2020,
            probability_predictions= rfc_probs_2020[:,1])

rfc_predictions_2020
rfc_predictions_2020.to_csv("rfc_predictions.csv")

##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Model 1.3 - Generalized Additive Models

from pygam import LogisticGAM


#Fit a GAM model with the default parameters
gam_model =  LogisticGAM()
gam_model.fit(X_train, y_train)


gam_pred_prob = gam_model.predict_proba(X_test)


gam_preds, complete_gam_dat = top_15_predictions(entire_test_data, gam_pred_prob )


gam_performance = all_nba_test_report(complete_gam_dat)


players_missed(complete_gam_dat)



gam_predict_probs_2020 = gam_model.predict_proba(features_2020)
gam_predict_binary_2020 = gam_model.predict(features_2020)

gam_predictions_2020 = predict_2020(positions=position_encoder.inverse_transform(features_2020['All_NBA_Pos']),
            player_names=current_dat['Player'],
            binary_prediction= gam_predict_binary_2020,
            probability_predictions= gam_predict_probs_2020)

gam_predictions_2020.to_csv("gam_predictions.csv")


##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Model 1.4 - KNN 
# --> Standardize all continuous features

# NOTE: We standardize the train set
# To standardize the test set, we use the mean and sd obtained from the train set (can't have leakage of info onto the test set)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train.drop('All_NBA_Pos',axis=1))

scaled_X_train = pd.DataFrame(scaler.transform(X_train.drop('All_NBA_Pos',axis=1)))
 
scaled_X_test = pd.DataFrame(scaler.transform(X_test.drop('All_NBA_Pos',axis=1)))



from sklearn.neighbors import KNeighborsClassifier

# How to choose value of k?
### --> Maybe use elbow method to choose k?
error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(scaled_X_train,y_train)
    pred_i = knn.predict(scaled_X_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

#plt.savefig('knn.png', bbox_inches='tight')

knn_model = KNeighborsClassifier(n_neighbors=11)

knn_model.fit(scaled_X_train,y_train)



knn_pred_prob = knn_model.predict_proba(scaled_X_test)


knn_preds, complete_knn_dat = top_15_predictions(entire_test_data, knn_pred_prob[:,1] )


knn_performance = all_nba_test_report(complete_knn_dat)


players_missed(complete_knn_dat)


# --> Predict 2020 season

# Standardize features
# --> Predict 2020 season

scaled_2020_dat = scaler.transform(current_dat[features_to_keep].drop('All_NBA_Pos',axis=1))

# Nice dataframe
scaled_features_2020 = pd.DataFrame(scaled_2020_dat)

#scaled_features_2020['All_NBA_Pos'] = position_encoder.fit_transform(current_dat['All_NBA_Pos'])
#scaled_features_2020
knn_predict_probs_2020 = knn_model.predict_proba(scaled_features_2020)
knn_predict_binary_2020 = knn_model.predict(scaled_features_2020)


knn_predictions_2020 = predict_2020(positions=position_encoder.inverse_transform(features_2020['All_NBA_Pos']),
            player_names=current_dat['Player'],
            binary_prediction= knn_predict_binary_2020,
            probability_predictions= knn_predict_probs_2020[:,1])


knn_predictions_2020.to_csv("knn_predictions.csv")
knn_predictions_2020





##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Model 1.5 - SVM
from sklearn.svm import SVC
# --> SVC = Support Vector Classifier
# --> C parameter allows you to choose how much you want to penalize misclassified points
#|--> Low C prioritizes simplicity (high bias, low variance)
#|--> High C may lead to overfitting

# --> kernel = rbf is the radial basis function
# --> If kernel = rbf, then we need an additional gamma parameter
# --> Basically LOW gamma goes for a simple fit, whereas a HIGH gamma goes for a complex fit

SVMmodel = SVC(C =1000, gamma = 0.001, probability = True, kernel = 'rbf')
SVMmodel.fit(scaled_X_train,y_train)


svm_pred_prob = SVMmodel.predict_proba(scaled_X_test)
svm_preds, complete_svm_dat = top_15_predictions(entire_test_data, svm_pred_prob[:,1] )

svm_performance = all_nba_test_report(complete_svm_dat)
players_missed(complete_knn_dat)


SVM_predict_probs_2020 = SVMmodel.predict_proba(scaled_features_2020)
SVM_predict_binary_2020 = SVMmodel.predict(scaled_features_2020)

svm_predictions_2020 = predict_2020(positions=position_encoder.inverse_transform(features_2020['All_NBA_Pos']),
            player_names=current_dat['Player'],
            binary_prediction= SVM_predict_binary_2020,
            probability_predictions= SVM_predict_probs_2020[:,1])
svm_predictions_2020.to_csv('svm_predictions.csv')
#
#### Grid search SVM
#param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001, 0.00001], 'kernel': ['rbf']} 
#from sklearn.model_selection import GridSearchCV
#grid = GridSearchCV(SVC(probability = True),param_grid,refit=True,verbose=3)
#grid.fit(scaled_X_train,y_train)
#grid.best_params_
#
## RANDOM SEARCH FOR 20 COMBINATIONS OF PARAMETERS
## DEFINE MODEL AND PERFORMANCE MEASURE
#
#from sklearn.model_selection import RandomizedSearchCV
#mdl = SVC(probability = True, random_state = 1)
#from scipy import stats
#rand_list = {"C": stats.uniform(2, 10),
#             "gamma": stats.uniform(0.1, 1)}
#
#rand_search = RandomizedSearchCV(mdl, param_distributions = rand_list, n_iter = 20, n_jobs = 4, cv = 3, random_state = 2017) 
#rand_search.fit(scaled_X_train,y_train) 
##rand_search.cv_results_
#
#svmrand_pred_prob = rand_search.predict_proba(scaled_X_test)
#svmrand_preds, complete_svmrand_dat = top_15_predictions(entire_test_data, svmrand_pred_prob[:,1] )
#all_nba_test_report(complete_svmrand_dat)
#players_missed(complete_knn_dat)
#
##SVM_predict_probs_2020 = rand_search.predict_proba(scaled_features_2020)
##SVM_predict_binary_2020 = rand_search.predict(scaled_features_2020)
#
##predict_2020(positions=position_encoder.inverse_transform(features_2020['All_NBA_Pos']),
##            player_names=current_dat['Player'],
##            binary_prediction= SVM_predict_binary_2020,
##            probability_predictions= SVM_predict_probs_2020[:,1])
#

#import pandas_profiling as pp



##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Model 1.6 - XGBoost
from xgboost import XGBClassifier


# --> Train model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict_proba(X_test)

xgb_preds, complete_xgb_dat = top_15_predictions(entire_test_data, xgb_predictions[:,1] )
xgb_performance = all_nba_test_report(complete_xgb_dat)
players_missed(complete_xgb_dat)



# --> xgboost feature importance
	
print(xgb_model.feature_importances_)


xgb_feature_dat = pd.DataFrame({'Importance': xgb_model.feature_importances_,
                                      'Feature': X_train.columns})

xgb_feature_importance = xgb_feature_dat .sort_values(by = ['Importance'], ascending=True).tail(30)


xgb_feature_importance['Feature']=np.where(xgb_feature_importance['Feature'] == 'ORtg', 'OFF Rating', 
         np.where(xgb_feature_importance['Feature'] == 'DRtg', 'DEF Rating',
                  np.where(xgb_feature_importance['Feature'] == 'Win_Prop', 'Team Performance',
                           np.where(xgb_feature_importance['Feature'] =="All_NBA_Pos", "Position",
                  xgb_feature_importance['Feature']))))


plt.figure(figsize=(10,6))
plt.barh(width = xgb_feature_importance['Importance'],
       y = xgb_feature_importance['Feature'], color = 'blue', alpha = 0.75)


plt.xlabel('Feature Importance Score', fontweight = "bold")
plt.ylabel('Features',fontweight = "bold")
plt.title("XGBoost  Feature Importance for\n All-NBA Selection",
          fontsize = 16, fontweight = "bold")

plt.show()



# --> xgboost 2020 predictions
xgb_predict_probs_2020 = xgb_model.predict_proba(features_2020)
xgb_predict_binary_2020 = xgb_model.predict(features_2020)

xgb_predictions_2020 = predict_2020(positions=position_encoder.inverse_transform(features_2020['All_NBA_Pos']),
            player_names=current_dat['Player'],
            binary_prediction= xgb_predict_binary_2020,
            probability_predictions= xgb_predict_probs_2020[:,1])


xgb_predictions_2020
xgb_predictions_2020.to_csv('xgb_predictions.csv')


##### ----- ##### ----- ##### ----- ##### -----# #### ----- ##### ----- ##### ----- ##### ----- #####
# Model 1.7 - Deep Learning



# Combine all results...



def get_votes(dat):
    '''
    Create new column where 1st team players awarded 5 votes,
    2nd team players 3 votes and 3rd team players 1 vote
    '''
    
    pred_alg = dat.copy()
    
    pred_alg['number'] = np.arange(0,dat.shape[0])
    
    first = np.array([0,5,6,13,14])
    second = np.array([1,7,8,15,16])
    third =  np.array([2,9,10,17,18])
    #reserves = np.array([3,4,11,12,19,20])
    
    pred_alg['votes'] = np.where(pred_alg['number'].isin(first), 3 , 
            np.where(pred_alg['number'].isin(second), 2,
             np.where(pred_alg['number'].isin(third), 1, 0.5)
             ))
    
    return pred_alg
    


df = pd.DataFrame()

results = [logreg_2020_predictions, rfc_predictions_2020, gam_predictions_2020, knn_predictions_2020, svm_predictions_2020]

for alg_results in results:
    df = df.append(get_votes(alg_results), ignore_index=True)


df_agg = df.groupby(['Position','Player']).sum()[['votes']]
df_agg.sort_values('votes', ascending = False)


votes_data = pd.DataFrame()
for position in ['Center', 'Forward', 'Guard']:
    sorted_dat = df_agg.iloc[df_agg.index.get_level_values('Position') == position].sort_values('votes', ascending = False)
    
    votes_data = votes_data.append(sorted_dat)

# --> Save data
votes_data.to_csv("votes_data.csv")


######################
# Aggregate the model performances

perf = {
        'All NBA Team':['1st', '2nd', '3rd', 'Not Selected'],
        'Logistic Regression': logreg_performance['binary_prediction'],
        'Random Forest': rfc_performance['binary_prediction'],
        'GAM': gam_performance['binary_prediction'],
        'KNN': knn_performance['binary_prediction'],
        'SVM': svm_performance['binary_prediction'],
        'XGB': xgb_performance['binary_prediction'],
        'True Total': [45,45,45,0]
        }



performances_dat = pd.DataFrame(perf)
performances_dat.to_csv("performances_dat.csv")