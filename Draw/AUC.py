# 2N-auc
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns  
%matplotlib inline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

attpdata  = pd.read_csv("attention_proba_test.csv", usecols = ["p2n", "p2y", "p3n", "p3y"])
attydata  = pd.read_csv("attention_proba_test.csv", usecols = ["label"])
attlabel = label_binarize(attydata, classes=[0, 1, 2, 3])
attpdata = np.array(attpdata)

svmpdata  = pd.read_csv("svm_proba_test.csv", usecols = ["p2n", "p2y", "p3n", "p3y"])
svmydata  = pd.read_csv("svm_proba_test.csv", usecols = ["label"])
svmlabel = label_binarize(svmydata, classes=[0, 1, 2, 3])
svmpdata = np.array(svmpdata)

lrpdata  = pd.read_csv("lr_proba_test.csv", usecols = ["p2n", "p2y", "p3n", "p3y"])
lrydata  = pd.read_csv("lr_proba_test.csv", usecols = ["label"])
lrlabel = label_binarize(lrydata, classes=[0, 1, 2, 3])
lrpdata = np.array(lrpdata)

rfpdata  = pd.read_csv("rf_proba_test.csv", usecols = ["p2n", "p2y", "p3n", "p3y"])
rfydata  = pd.read_csv("rf_proba_test.csv", usecols = ["label"])
rflabel = label_binarize(rfydata, classes=[0, 1, 2, 3])
rfpdata = np.array(rfpdata)


# 计算每一类的ROC
n_classes = 4
attfpr = dict()
atttpr = dict()
attroc_auc = dict()
for i in range(n_classes):
    attfpr[i], atttpr[i], _ = roc_curve(attlabel[:, i], attpdata[:, i])
    attroc_auc[i] = auc(attfpr[i], atttpr[i])
# Compute micro-average ROC curve and ROC area
attfpr["micro"], atttpr["micro"], _ = roc_curve(attlabel.ravel(), attpdata.ravel())
attroc_auc["micro"] = auc(attfpr["micro"], atttpr["micro"])

n_classes = 4
svmfpr = dict()
svmtpr = dict()
svmroc_auc = dict()
for i in range(n_classes):
    svmfpr[i], svmtpr[i], _ = roc_curve(svmlabel[:, i], svmpdata[:, i])
    svmroc_auc[i] = auc(svmfpr[i], svmtpr[i])
# Compute micro-average ROC curve and ROC area
svmfpr["micro"], svmtpr["micro"], _ = roc_curve(svmlabel.ravel(), svmpdata.ravel())
svmroc_auc["micro"] = auc(svmfpr["micro"], svmtpr["micro"])

n_classes = 4
rffpr = dict()
rftpr = dict()
rfroc_auc = dict()
for i in range(n_classes):
    rffpr[i], rftpr[i], _ = roc_curve(rflabel[:, i], rfpdata[:, i])
    rfroc_auc[i] = auc(rffpr[i], rftpr[i])
# Compute micro-average ROC curve and ROC area
rffpr["micro"], rftpr["micro"], _ = roc_curve(rflabel.ravel(), rfpdata.ravel())
rfroc_auc["micro"] = auc(rffpr["micro"], rftpr["micro"])

n_classes = 4
lrfpr = dict()
lrtpr = dict()
lrroc_auc = dict()
for i in range(n_classes):
    lrfpr[i], lrtpr[i], _ = roc_curve(lrlabel[:, i], lrpdata[:, i])
    lrroc_auc[i] = auc(lrfpr[i], lrtpr[i])
# Compute micro-average ROC curve and ROC area
lrfpr["micro"], lrtpr["micro"], _ = roc_curve(lrlabel.ravel(), lrpdata.ravel())
lrroc_auc["micro"] = auc(lrfpr["micro"], lrtpr["micro"])


sns.set(style="ticks", palette="muted", color_codes=True, font="Times New Roman")
#sns.set_style("grid")  # "white", "dark", "whitegrid", "darkgrid", "ticks"
plt.figure(figsize=(8, 8))
plt.grid()
roc_auc_2N = [svmroc_auc[0], rfroc_auc[0], lrroc_auc[0], attroc_auc[0]]
fpr_2N = [svmfpr[0], rffpr[0], lrfpr[0], attfpr[0]]
tpr_2N = [svmtpr[0], rftpr[0], lrtpr[0], atttpr[0]]


colors = cycle(['aqua', 'darkorange', 'seagreen', "darkblue"])#cornflowerblue
name =  ['SVM', 'RF', 'LR', "SA-BiLSTM"]
for i, color in zip(range(4), colors):
    plt.plot(fpr_2N[i], tpr_2N[i], color=color,
             label='{0} (AUC = {1:0.3f})'
             ''.format(name[i], roc_auc_2N[i]),  linewidth=2)
    
plt.plot([0, 1], [0, 1], '-', color='pink', linewidth=1)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])

frame = plt.gca()
frame.spines["top"].set_visible(False)
frame.spines["right"].set_visible(False)
plt.tick_params(which='major', left=True,bottom=True, colors='black', width=1,length=8, direction='out')
plt.tick_params(labelsize=18)
plt.xlabel('False Positive Rate',  fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
#plt.title('Traing Set')
plt.legend(loc="lower right",  fontsize=18)# loc='best' loc="lower right"
plt.show()
#plt.savefig('2N-auc-test',dpi = 1200)

sns.set(style="ticks", palette="muted", color_codes=True, font="Times New Roman")
#sns.set_style("grid")  # "white", "dark", "whitegrid", "darkgrid", "ticks"
plt.figure(figsize=(8, 8))
plt.grid()
roc_auc_2Y = [svmroc_auc[1], rfroc_auc[1], lrroc_auc[1], attroc_auc[1]]
fpr_2Y = [svmfpr[1], rffpr[1], lrfpr[1], attfpr[1]]
tpr_2Y = [svmtpr[1], rftpr[1], lrtpr[1], atttpr[1]]


colors = cycle(['aqua', 'darkorange', 'seagreen', "darkblue"])#cornflowerblue
name =  ['SVM', 'RF', 'LR', "SA-BiLSTM"]
for i, color in zip(range(4), colors):
    plt.plot(fpr_2Y[i], tpr_2Y[i], color=color,
             label='{0} (AUC = {1:0.3f})'
             ''.format(name[i], roc_auc_2Y[i]),  linewidth=2)
    
plt.plot([0, 1], [0, 1], '-', color='pink', linewidth=1)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])

frame = plt.gca()
frame.spines["top"].set_visible(False)
frame.spines["right"].set_visible(False)
plt.tick_params(which='major', left=True,bottom=True, colors='black', width=1,length=8, direction='out')
plt.tick_params(labelsize=18)
plt.xlabel('False Positive Rate',  fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
#plt.title('Traing Set')
plt.legend(loc="lower right",  fontsize=18)# loc='best' loc="lower right"
#plt.show()
plt.savefig('2Y-auc-test',dpi = 1200)

sns.set(style="ticks", palette="muted", color_codes=True, font="Times New Roman")
#sns.set_style("grid")  # "white", "dark", "whitegrid", "darkgrid", "ticks"
plt.figure(figsize=(8, 8))
plt.grid()
roc_auc_3N = [svmroc_auc[2], rfroc_auc[2], lrroc_auc[2], attroc_auc[2]]
fpr_3N = [svmfpr[2], rffpr[2], lrfpr[2], attfpr[2]]
tpr_3N = [svmtpr[2], rftpr[2], lrtpr[2], atttpr[2]]


colors = cycle(['aqua', 'darkorange', 'seagreen', "darkblue"])#cornflowerblue
name =  ['SVM', 'RF', 'LR', "SA-BiLSTM"]
for i, color in zip(range(4), colors):
    plt.plot(fpr_3N[i], tpr_3N[i], color=color,
             label='{0} (AUC = {1:0.3f})'
             ''.format(name[i], roc_auc_3N[i]),  linewidth=2)
    
plt.plot([0, 1], [0, 1], '-', color='pink', linewidth=1)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])

frame = plt.gca()
frame.spines["top"].set_visible(False)
frame.spines["right"].set_visible(False)
plt.tick_params(which='major', left=True,bottom=True, colors='black', width=1,length=8, direction='out')
plt.tick_params(labelsize=18)
plt.xlabel('False Positive Rate',  fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
#plt.title('Traing Set')
plt.legend(loc="lower right",  fontsize=18)# loc='best' loc="lower right"
#plt.show()
plt.savefig('3N-auc-test',dpi = 1200)

sns.set(style="ticks", palette="muted", color_codes=True, font="Times New Roman")
#sns.set_style("grid")  # "white", "dark", "whitegrid", "darkgrid", "ticks"
plt.figure(figsize=(8, 8))
plt.grid()
roc_auc_3Y = [svmroc_auc[3], rfroc_auc[3], lrroc_auc[3], attroc_auc[3]]
fpr_3Y = [svmfpr[3], rffpr[3], lrfpr[3], attfpr[3]]
tpr_3Y = [svmtpr[3], rftpr[3], lrtpr[3], atttpr[3]]


colors = cycle(['aqua', 'darkorange', 'seagreen', "darkblue"])#cornflowerblue
name =  ['SVM', 'RF', 'LR', "SA-BiLSTM"]
for i, color in zip(range(4), colors):
    plt.plot(fpr_3Y[i], tpr_3Y[i], color=color,
             label='{0} (AUC = {1:0.3f})'
             ''.format(name[i], roc_auc_3Y[i]),  linewidth=2)
    
plt.plot([0, 1], [0, 1], '-', color='pink', linewidth=1)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])

frame = plt.gca()
frame.spines["top"].set_visible(False)
frame.spines["right"].set_visible(False)
plt.tick_params(which='major', left=True,bottom=True, colors='black', width=1,length=8, direction='out')
plt.tick_params(labelsize=18)
plt.xlabel('False Positive Rate',  fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
#plt.title('Traing Set')
plt.legend(loc="lower right",  fontsize=18)# loc='best' loc="lower right"
#plt.show()
plt.savefig('3Y-auc-test',dpi = 1200)

sns.set(style="ticks", palette="muted", color_codes=True, font="Times New Roman")
#sns.set_style("grid")  # "white", "dark", "whitegrid", "darkgrid", "ticks"
plt.figure(figsize=(8, 8))
plt.grid()
roc_auc = [svmroc_auc["micro"], rfroc_auc["micro"], lrroc_auc["micro"], attroc_auc["micro"]]
fpr = [svmfpr["micro"], rffpr["micro"], lrfpr["micro"], attfpr["micro"]]
tpr = [svmtpr["micro"], rftpr["micro"], lrtpr["micro"], atttpr["micro"]]


colors = cycle(['aqua', 'darkorange', 'seagreen', "darkblue"])#cornflowerblue
name =  ['SVM', 'RF', 'LR', "SA-BiLSTM"]
for i, color in zip(range(4), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label='{0} (AUC = {1:0.3f})'
             ''.format(name[i], roc_auc[i]),  linewidth=2)
    
plt.plot([0, 1], [0, 1], '-', color='pink', linewidth=1)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])

frame = plt.gca()
frame.spines["top"].set_visible(False)
frame.spines["right"].set_visible(False)
plt.tick_params(which='major', left=True,bottom=True, colors='black', width=1,length=8, direction='out')
plt.tick_params(labelsize=18)
plt.xlabel('False Positive Rate',  fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
#plt.title('Traing Set')
plt.legend(loc="lower right",  fontsize=18)# loc='best' loc="lower right"
plt.show()
#plt.savefig('Average-auc-test',dpi = 1200)
