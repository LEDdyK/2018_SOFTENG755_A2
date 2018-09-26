
import numpy as np
from timeit import default_timer as timer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import pandas as pd

random_seed = 5432363

def writeData(src, train, test):
    train.reset_index().to_csv(src + '/train.csv', index=False)
    test.reset_index().to_csv(src + '/test.csv', index=False)

# World Cup 2018 Data Preparation
dat1 = pd.read_csv("data/World_Cup_2018/2018 worldcup.csv")
dat1.Total_Scores.describe()
dat1['IsNormalTime'] = 1
dat1.loc[dat1.Normal_Time == 'No', 'IsNormalTime'] = 0
dat1['indices'] = 5
dat1.loc[dat1.Match_ID < 61, 'indices'] = 4
dat1.loc[dat1.Match_ID < 57, 'indices'] = 3
dat1.loc[dat1.Match_ID < 49, 'indices'] = 2
dat1.loc[dat1.Match_ID < 33, 'indices'] = 1
dat1.loc[dat1.Match_ID < 17, 'indices'] = 0
dat1.indices.value_counts() # world cup match cycle
dat1.drop(columns=['Match_result', 'Team1_Continent', 'Team2_Continent', 'Normal_Time','Phase', 'Location', 'Date'], inplace=True) # Phase, Location and Continent can perform further aggregation

match_result = dat1[['IsNormalTime','Total_Scores','indices']]
match_result.columns = ['IsNormalTime','Scores','indices']
cols_name = ['Team', 'Attempts', 'Corners', 'Offsides', 
        'Ball_Possession(%)', 'Pass_Accuracy(%)', 'Distance_Covered', 
        'Ball_Recovered', 'Yellow_Card', 'Red_Card', 'Team1_Fouls']
team1 = dat1.loc[:, dat1.columns[1:12]]
team1.columns = cols_name
team1_result = pd.concat([match_result, team1],axis=1)
team2 = dat1.loc[:, dat1.columns[12:23]]
team2.columns = cols_name
team2_result = pd.concat([match_result, team2],axis=1)
all_team = pd.concat([team1_result, team2_result], axis=0)
all_team.reset_index(inplace=True, drop=True)
all_team.sort_values(['Team', 'indices'],ascending=True, inplace=True)
all_team.reset_index(inplace=True, drop=True)
all_team.to_csv("data/World_Cup_2018/statistics.csv", index=False)
schedule = dat1[['Team1', 'Team2','Total_Scores','indices']]
test = schedule.query('indices >= 5').reset_index(drop=True)
train = schedule.query('indices < 5').reset_index(drop=True)
writeData('data/World_Cup_2018', train, test)

# Traffic Congestion Data Preparation
dat2 = pd.read_csv("data/Traffic_flow/traffic_flow_data.csv")
train, test = train_test_split(dat2, test_size=0.1, random_state=random_seed)
writeData('data/Traffic_flow', train, test)

# Occupancy Sensor Data Preparation
dat3 = pd.read_csv("data/Occupancy_sensor/occupancy_sensor_data.csv")
dat3.Occupancy.value_counts() # classification
dat3.sort_values('date', inplace=True)
dat3.reset_index(inplace=True)
# parse datetime
dat3.date = pd.to_datetime(dat3.date)

# extract day
dat3['day'] = dat3.date.dt.date

# group by rank
dat3['id'] = dat3.groupby('day')['index'].rank(ascending=True)

dat3.drop(['date', 'index'],axis=1,inplace=True)

# base data
base = dat3[['day', 'id', 'Occupancy']]
measurement = dat3.copy()
measurement.drop('Occupancy', axis=1, inplace=True) # ensure measurement has no occupancy information
measurement.sort_values(['day', 'id'], inplace=True)
Ks = [0, 1, 3, 5, 7, 9, 11]
for index, k in enumerate(Ks):
    temp1 = measurement.copy() 
    temp1['id'] = temp1['id'] - k
    temp1 = temp1.query('id > 0') # temp1 is before
    temp2 = measurement.merge(temp1[['day', 'id']], on=['day', 'id'], how='inner') # temp2 is current
    temp1.reset_index(inplace=True, drop=True)
    temp2.reset_index(inplace=True, drop=True)
    difft = temp1.iloc[:,0:5] - temp2.iloc[:, 0:5]
    difft.columns = ['diff_' + item for item in difft.columns]
    temp3 = temp1[['day', 'id']]
    temp4 = pd.concat([temp3, difft],axis=1)
    temp4['id'] = temp4['id'] + k
    dat3_new = dat3.merge(temp4, how='left', on=['day', 'id'])
    dat3_new = dat3_new.fillna(0)
    train, test = train_test_split(dat3_new, test_size=0.1, random_state=random_seed)
    train.reset_index().to_csv('data/Occupancy_sensor/train_%s.csv' % index, index=False)
    test.reset_index().to_csv('data/Occupancy_sensor/test_%s.csv' % index, index=False)


# LandSat Data Preparation
# In each line of data the four spectral values for the top-left pixel are given first 
# followed by the four spectral values for the top-middle pixel 
# and then those for the top-right pixel, so on with the pixels read 
# out in sequence left-to-right and top-to-bottom. 
# Thus, the four spectral values for the central pixel are given by attributes 17,18,19 and 20. 
# If you like you use only these four attributes, 
# while ignoring the others. 
# This avoids the problem which arises when a 3x3 neighbourhood straddles a boundary.
location = ["top-left", "top-mid", "top-right",
            "mid-left", "mid-mid", "mid-right",
            "bottom-left","bottom-mid", "bottom-right"]
location2 = [item+"_"+ str(num) for item in location for num in [1,2,3,4]]
dat4 = pd.read_csv("data/Landsat/lantsat.csv", names=location2+["label"])
dat4.label.value_counts() # classification
dat4.iloc[:,16:20] # core
train, test = train_test_split(dat4, test_size=0.1, stratify = dat4.label, random_state=random_seed)
writeData('data/Landsat', train, test)

