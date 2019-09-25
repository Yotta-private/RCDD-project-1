import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv("prob-1234.csv")
#data.head(3)
prob = data[['p2n','p2y', "p3n", "p3y"]]
pred = list(data["predlabel"])
type_2 = list(data["acturaltarget"])

cv_1_test = np.array(pred)
Y_1 = np.array(type_2)
prob_test = np.array(prob)

print("Test_accuracy", accuracy_score(Y_1, cv_1_test))
test_binarize = label_binarize(Y_1, classes=[0, 1, 2, 3])
#prob_test = prob_test[1:]
print("Test AUC:", roc_auc_score(test_binarize, prob_test, average = 'macro'))
print("Test cohen_kappa_score:", cohen_kappa_score(Y_1, cv_1_test))
print("Test hamming_loss:", hamming_loss(Y_1, cv_1_test))
target_names = ['2N', '2Y', '3N', "3Y"]
print("Test classification_report:", classification_report(Y_1, cv_1_test, target_names=target_names, digits=4)) # avg / total加权平均
print ("Test precision_score_macro:", precision_score(Y_1, cv_1_test,average='macro')) # precision
print("Test recall_score_macro:", recall_score(Y_1, cv_1_test,average='macro')) # recall
#print(accuracy_score(y_test, test_pred, normalize=False)) # correct
print("Test confusion_matrix:", confusion_matrix(Y_1, cv_1_test)) 
