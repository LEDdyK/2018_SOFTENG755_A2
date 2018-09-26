import time as time
import numpy as np
from __future__ import print_function
from timeit import default_timer as timer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,  Normalizer

import pandas as pd

random_seed = 5432363

# Load data
train = pd.read_csv("data/World_Cup_2018/train.csv")
test = pd.read_csv("data/World_Cup_2018/test.csv")
stats = pd.read_csv("data/World_Cup_2018/statistics.csv")

del train['index']
del test['index']


# Normalization/Standardization
col_scalling = {'Identity':None ,'ZScore':  StandardScaler, 'MinMax': MinMaxScaler}

# Feature Extraction
def extract(data):
    tmp = data.copy()
    y = data.Total_Scores.tolist()
    tmp.drop(columns=['index', 'Team1', 'Team2','Total_Scores'], inplace=True)
    x = tmp
    return y, x

def extract_stats(data, stat, inds):
    output = []
    for ind in inds:
        target_data = data.query("indices == " + str(ind))
        target_data.reset_index(inplace=True)
        target_stats0 = stat.query("indices < " + str(ind))
        del target_stats0['indices']
        target_stats1 = target_stats0.groupby(['Team']).agg(['mean'])
        target_stats1.reset_index(inplace=True)
        target_stats1.columns = [item[0] for item in target_stats1.columns ]
        
        team1_0 = target_stats1.merge(target_data[['Team1','index']], right_on='Team1', left_on='Team')
        del team1_0['Team1']
        team1_0.sort_values('index', inplace=True)
        team2_0 = target_stats1.merge(target_data[['Team2','index']], right_on='Team2', left_on='Team')
        del team2_0['Team2']
        team2_0.sort_values('index', inplace=True)
        team1_1 = team1_0.iloc[:,1:-1].reset_index(drop=True)
        team2_1 = team2_0.iloc[:,1:-1].reset_index(drop=True)
        diff = abs(team1_1 - team2_1)
        out0 = pd.concat([target_data, diff],axis=1)
        output.append(out0)
    out = pd.concat(output, axis=0)
    return out


# Cross Validation
tscv = TimeSeriesSplit(n_splits=4)
X = np.array(range(1,6))
ts_split = []
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    print(X[train_index])
    print(X[test_index])
    ts_split.append((X[train_index].tolist(), X[test_index].tolist()))
    

# Hyperparameter Space
tuned_parameters = {
        "Ridge": [{'fit_intercept': [True, False], 'normalize': [True, False], 'alpha': np.logspace(0, -5, num=3)}],
        "LM" : [{"fit_intercept": [True, False], 'normalize': [True, False]}]
}
score = 'neg_median_absolute_error' # 

# Models
models = {
    "Ridge": Ridge(solver='lsqr', max_iter = 9999), 
    "LM": LinearRegression()
}

# Matrix
cv_result = []
for mkey, model in models.iteritems():
    print("# Staring fittimg model of %s" % mkey)
    print()
    print("# Tuning hyper-parameters for %s" % score)
    print()
    d = tuned_parameters[mkey][0]
    pl = [dict(zip(d, v)) for v in product(*d.values())]
    for ps in pl:
        model.set_params(**ps)
        y_true = []
        y_cv = []
        for ts_i in ts_split[:-1]:
            train1 = extract_stats(train, stats, ts_i[0])
            val1 = extract_stats(train, stats, ts_i[1])
            y_train, X_train1 = extract(train1)
            y_val, X_val1 = extract(val1)
            print("Train: " + str(ts_i[0]))
            print(X_train1.shape)
            print("Validation: "+ str(ts_i[1]))
            print(X_val1.shape)
            model.fit(X_train1.values, y_train)
            print(len(y_val))
            y_true = y_true + y_val
            y_cv = y_cv + model.predict(X_val1.values).tolist()
        res = median_absolute_error(y_true, y_cv)
        print(res)
        cv_result.append((mkey, ps, res, len(y_true)))

cv_dat = pd.DataFrame(cv_result, columns=['Method','Params', 'CV', 'NumberOfCV'])
cv_dat.sort_values(['Method', 'CV'], ascending=True, inplace=True)
cv_dat.reset_index(inplace=True, drop=True)
cv_dat.to_csv("data/World_Cup_2018/cv_result.csv", index=False)
or_dat = cv_dat.query('Method == "LM"').iloc[0,:]
rr_dat = cv_dat.query('Method == "Ridge"').iloc[0,:]
fcv = pd.DataFrame([or_dat, rr_dat])


# apply to test set
test_result = []
for index, row in fcv.iterrows():
    for ts_i in [ts_split[-1]]:
        train1 = extract_stats(train, stats, ts_i[0])
        test1 = extract_stats(test, stats, ts_i[1])
        y_train, X_train1 = extract(train1)
        y_test, X_test1 = extract(test1)
        print("Train: " + str(ts_i[0]))
        print(X_train1.shape)
        print("Validation: "+ str(ts_i[1]))
        print(X_val1.shape)
        best_model = models[row['Method']]
        best_model.set_params(**row['Params'])
        best_model.fit(X_train1.values, y_train)
        coef_dat = pd.DataFrame(best_model.coef_, columns=['coef'])
        coef_dat['vars'] = X_train1.columns
        coef_dat.to_csv("data/World_Cup_2018/best_%s.csv" % row['Method'],index=False)
        y_true = y_test
        y_pred = best_model.predict(X_test1.values).tolist()
        final_score = mean_squared_error(y_true, y_pred)
        print(final_score)
        test_result.append((row['Method'], row['Params'], final_score, len(y_true)))

test_output = pd.DataFrame(test_result,columns=['Method', 'Params', 'TestScore', 'NumberOfTestCases'])
test_output.to_csv("data/World_Cup_2018/ho_result.csv", index=False)


