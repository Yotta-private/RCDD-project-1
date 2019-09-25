
import numpy as np
import pandas as pd
from  kfold_ratio import *
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
%matplotlib inline

def load_data(filename):
    data = pd.read_csv(filename)
    data.drop(data.columns[[0,1,2,3]], axis=1,inplace=True)
    return data

df = load_data("RAC_log_des.csv") # 加载描述符csv文件 Y=1

lgb_params = dict(
    boosting_type = "gbdt",
    objective = "binary",
    metric = "binary_logloss",
    learning_rate = 0.005,
    max_depth = -1,
    num_leaves = 200,
    max_bin = 250,
    colsample_bytree = 0.5,
    reg_lambda = 2.8,
    min_data_in_leaf = 20)

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
cv_test_pred = []
Y_label_test = []
cv_train_pred = []
Y_label_train = []

for i, (train_idx, test_idx) in enumerate(kf.split(df)):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    #print(len(train_idx))
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    y_train = train["type"]
    #print(y_train)
    #y_train = np.array(y_train).reshape(y_train.shape[0], 1) 
    y_test = test["type"]
    #y_test = np.array(y_test).reshape(y_test.shape[0], 1)
    #print(train.head(3))
    model_x_train = train
    model_x_test =   test
    model_x_train.drop(model_x_train.columns[[0,1]], axis=1,inplace=True)
    x_train = np.array(model_x_train)
    model_x_test.drop(model_x_test.columns[[0,1]], axis=1,inplace=True)
    x_test = np.array(model_x_test)
    
    # processing data
    x_data = np.vstack((x_train,x_test))
    scaler = StandardScaler().fit(x_data)
    x_train =  scaler.transform(x_train)
    x_test =  scaler.transform(x_test)
    
    # model trainning
    gbm_train = lgb.Dataset(model_x_train, y_train)
    gbm_test = lgb.Dataset(model_x_test, y_test)
    
    gbm = lgb.train(lgb_params,
                         gbm_train,
                         valid_sets=gbm_test,
                         num_boost_round=1000,
                         early_stopping_rounds=80,
                         verbose_eval=-1)
    
    
    train_pred = gbm.predict(x_train)
    # print(train_pred) binary-loss
    train_pred = np.where(train_pred>=0.5, 1, 0)
    #print(train_pred)   
    test_pred = gbm.predict(x_test)
    test_pred = np.where(test_pred>=0.5, 1, 0)
    cv_test_pred.append(test_pred)
    Y_label_test.append(y_test)
    cv_train_pred.append(train_pred)
    Y_label_train.append(y_train)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    print("train_accuracy", train_accuracy)
    print("Train AUC:", roc_auc_score(y_train, train_pred))
    print("train classification_report:", classification_report(y_train, train_pred))
    print("--------------------------------------------------------------------------------------")
    print("test_accuracy", test_accuracy)
    print("Test AUC:", roc_auc_score(y_test, test_pred))
    print("test classification_report:", classification_report(y_test, test_pred))
print("****************************************************************")
#print(cv_test_pred)
#print(Y_label_test)
cv_test_pred = [y for x in cv_test_pred for y in x]
y_label_test = []
for i in range(len(Y_label_test)):
    label = list(Y_label_test[i])
    y_label_test.append(label)
y_label_test = [y for x in y_label_test for y in x]
print("Test Accuracy", accuracy_score(y_label_test, cv_test_pred))
print("Test AUC:", roc_auc_score(y_label_test, cv_test_pred))
print("Test report:", classification_report(y_label_test, cv_test_pred))
print("****************************************************************")
cv_train_pred = [y for x in cv_train_pred for y in x]
y_label_train = []
for i in range(len(Y_label_train)):
    label = list(Y_label_train[i])
    y_label_train.append(label)
y_label_train = [y for x in y_label_train for y in x]
print("Train Accuracy", accuracy_score(y_label_train, cv_train_pred))
print("Train AUC:", roc_auc_score(y_label_train, cv_train_pred))
print("Train report:", classification_report(y_label_train, cv_train_pred))
