"*****************************************************"
# Code for classification by RF
# Descriptors : RAC_ecfp10_share
# split method : random
"*****************************************************"

import numpy as np
import pandas as pd
from  kfold_ratio import *
from sklearn.ensemble import RandomForestClassifier
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


RF_model = RandomForestClassifier()

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
    y_train = np.array(y_train).reshape(y_train.shape[0], 1) 
    y_test = test["type"]
    y_test = np.array(y_test).reshape(y_test.shape[0], 1)
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
    RF_model .fit(x_train, y_train.ravel())
    
    
    train_pred = RF_model.predict(x_train)
    test_pred = RF_model.predict(x_test)
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
cv_test_pred = [y for x in cv_test_pred for y in x]
c1 = [x for x in Y_label_test]
c2 = [x.flatten() for x in c1]
Y_label_test = [y for x in c2 for y in x]
print("Test Accuracy", accuracy_score(Y_label_test, cv_test_pred))
print("Test AUC:", roc_auc_score(Y_label_test, cv_test_pred))
print("Test report:", classification_report(Y_label_test, cv_test_pred))
print("****************************************************************")
cv_train_pred = [y for x in cv_train_pred for y in x]
c1 = [x for x in Y_label_train]
c2 = [x.flatten() for x in c1]
Y_label_train = [y for x in c2 for y in x]
print("Train Accuracy", accuracy_score(Y_label_train, cv_train_pred))
print("Train AUC:", roc_auc_score(Y_label_train, cv_train_pred))
print("Train report:", classification_report(Y_label_train, cv_train_pred))
