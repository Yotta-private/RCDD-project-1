# 基于sklearn.model_selection.GridSearchCV 调参
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def load_data(filename):
    "load data without label"
    with open(filename) as f:
        ncols = len(f.readline().split(','))
    data = pd.read_csv(filename, usecols = range(1, ncols - 2)) # without the 'name' column
    "load label"
    label = pd.read_csv(filename, usecols = ['yield'])
    return data, label

data_train = load_data('RAC_train_cut_n3.csv')
X_train = data_train[0]
y_train = data_train[1]
data_test = load_data('RAC_test_cut_n3.csv')
X_test = data_test[0]
y_test = data_test[1]

print(X_train.head(3))
print('---------------------')
print(X_test.head(3))

# data preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train) #numpy
X_test = scaler.transform(X_test)
print(X_train[0:5])
print('----------------------')
print(X_test[0:5])

y_train = np.array(y_train)
# y_train = y_train.tolist()
y_test = np.array(y_test)
# y_test = y_test.tolist()
print(X_train.shape)
print(X_test.shape)
print('----------------------')
print(y_train.shape)
print(y_test.shape)

"""# model defination
# specify parameters
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
param = {'max_depth':5,
         'eta':0.1,
         'silent':1,
         'objective':'reg:linear',
         'eval_metric':['rmse']} # 'auc'？？？error？？？？？
# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 100
evals_result = {}

#train_model
bst = xgb.train(param, dtrain, num_round, watchlist, evals_result=evals_result)

from sklearn.metrics import r2_score
preds_train = bst.predict(dtrain)
R2_train = r2_score(dtrain.get_label(), preds_train)
print (R2_train)

preds_test = bst.predict(dtest)
R2_test = r2_score(dtest.get_label(), preds_test)
print (R2_test)"""

# plot result
import matplotlib.pyplot as plt

#开启一个窗口，num设置子图数量，这里如果在add_subplot里写了子图数量，num设置多少就没影响了
#figsize设置窗口大小，dpi设置分辨率
fig = plt.figure(num=2, figsize=(15, 15),dpi=700)

#使用add_subplot在窗口加子图，其本质就是添加坐标系
#三个参数分别为：行数，列数，本子图是所有子图中的第几个，最后一个参数设置错了子图可能发生重叠
ax1 = fig.add_subplot(2,1,1)  
ax2 = fig.add_subplot(2,1,2)

#绘制直线
x=np.linspace(0,80,100)
y=x
ax1.plot(x,y,color='gray')
#绘制曲线
x1 = dtrain.get_label()
y1 = preds_train
ax1.scatter(x1, y1, marker = '*', color = 'blue', label='1', s = 10)

#绘制直线
ax2.plot(x,y,color='gray')
#绘制曲线
x2 = dtest.get_label()
y2 = preds_test
ax2.scatter(x2, y2, marker = '+', color = 'green', label='2', s = 10)
plt.show()

# 最佳迭代次数：n_estimators
if __name__ == '__main__':

    cv_params = {'n_estimators': [400, 500, 600, 700, 800, 900]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=6)
    optimized_GBM.fit(X_train, y_train)
    # evalute_result = optimized_GBM.grid_scores_
    means = optimized_GBM.cv_results_['mean_test_score']
    print('每轮迭代运行结果:{0}'.format(means))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

from numpy import sort
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

# fit model on all training data
model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=585, max_depth=3, min_child_weight=7, seed=0,
                             subsample=0.8, colsample_bytree=0.8, gamma=0.5, reg_alpha=0.1, reg_lambda=3)
model.fit(X_train, y_train)
# make predictions for test data and evaluate
# y_pred = model.predict(X_test)

thresholds = sort(model.feature_importances_)
# print(thresholds)
for thresh in thresholds:
    if thresh > 0:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=585, max_depth=3, min_child_weight=7, seed=0,
                                 subsample=0.8, colsample_bytree=0.8, gamma=0.5, reg_alpha=0.1, reg_lambda=3)
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        pred_select_test = selection_model.predict(select_X_test)
        R2_select_test = r2_score(y_test,  pred_select_test)
        # print(R2_select_test)
        print("Thresh=%.9f, n=%d, R2: %.2f%%" % (thresh, select_X_train.shape[1], R2_select_test*100.0))

thresholds = sort(model.feature_importances_)
# print(thresholds)
for thresh in thresholds:
    print(thresh)
