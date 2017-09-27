import numpy as np
import pandas as pd

file = r'E:\p6M_mdl.csv'
reader = pd.read_csv(file, header=0, index_col=None, chunksize=10000)
p6m = pd.DataFrame()
for chunk in reader:
    chunk['pyear_month'] = chunk['pyear_month'].str.split(':').str.get(0)
    chunk['pyear_month'] = pd.to_datetime(chunk['pyear_month'],format='%d%b%y')
    chunk = chunk[chunk.pyear_month==pd.datetime(2016,10,1)]
    p6m = pd.concat([p6m,chunk])
   
dep = pd.read_csv(r'E:\dep_mdl.csv', header=0, index_col=None, usecols=[0,1])
#join data according to user ID
train = pd.merge(p6m, dep, how='inner', on=['fuid_md5'])
#fcredit_update_time
train['fcredit_update_time'] = train['fcredit_update_time'].str.split(':').str.get(0)
train['fcredit_update_time'] = pd.to_datetime(train['fcredit_update_time'],format='%d%b%y')
train['fcredit_update_time'] = (pd.datetime(2016,10,1)-train['fcredit_update_time'])/ pd.Timedelta(days=1)
#cyc_date
train['cyc_date'] = pd.to_datetime(train['cyc_date'],format='%Y-%m-%d')
train['cyc_date'] = train['cyc_date'].dt.day

#xgboost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import train_test_split  #Train_test_split

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def modelfit(alg, dtrain, predictors,target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,\
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print("n_estimators : %d" % cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #train_test_split
    train, test = train_test_split(dtrain, train_size=0.75, random_state=1)
    #Fit the algorithm on the data
    alg.fit(train[predictors], train[target], eval_metric='auc')
        
    #Predict training set:
    #train_predictions = alg.predict(train[predictors])
    train_predprob = alg.predict_proba(train[predictors])[:,1]
    #Predict testing set:
    test_predictions = alg.predict(test[predictors])
    test_predprob = alg.predict_proba(test[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    #print ("Accuracy : %.4g" % metrics.accuracy_score(test[target].values, test_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(train[target], train_predprob))
    print ("AUC Score (Test): %f" % metrics.roc_auc_score(test[target], test_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
#Choose all predictors except target
target = 'dep'
predictors = [x for x in train.columns if x not in ['fuid_md5','pyear_month',target]]
xgb1 = XGBClassifier(learning_rate =0.5,n_estimators=1000,max_depth=5,min_child_weight=1,\
    gamma=0,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',nthread=4,\
    scale_pos_weight=1,seed=27) #scale_pos_weight
modelfit(xgb1, train, predictors, target)
