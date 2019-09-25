import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

pdata  = pd.read_csv("attention_proba_train.csv", usecols = ["p2n", "p2y", "p3n", "p3y"])
ydata  = pd.read_csv("attention_proba_train.csv", usecols = ["label"])
label = label_binarize(ydata, classes=[0, 1, 2, 3])
pdata = np.array(pdata)
#print(roc_auc_score(label , pdata))

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 4
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(label[:, i], pdata[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(label.ravel(), pdata.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

import seaborn as sns  
%matplotlib inline
#cm = plt.cm.get_cmap('RdYlBu')
sns.set(style="ticks", palette="muted", color_codes=True, font="Times New Roman")
#sns.set_style("grid")  # "white", "dark", "whitegrid", "darkgrid", "ticks"
plt.figure(figsize=(8, 8))
plt.grid()
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro (AUC = {0:0.3f})'
               ''.format(roc_auc["micro"]),
         color='gold', linestyle='-', linewidth=2)

"""plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (AUC = {0:0.3f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)"""

colors = cycle(['aqua', 'darkorange', 'seagreen', "darkblue"])#cornflowerblue
name =  ['2N', '2Y', '3N', "3Y"]
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label='Class {0} (AUC = {1:0.3f})'
             ''.format(name[i], roc_auc[i]),  linewidth=2)
plt.plot([0, 1], [0, 1], '-', color='pink', linewidth=1)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])

frame = plt.gca()
#frame.axes.get_yaxis().set_visible(True)
#frame.axes.get_xaxis().set_visible(True)
frame.spines["top"].set_visible(False)
frame.spines["right"].set_visible(False)
plt.tick_params(which='major', left=True,bottom=True, colors='black', width=1,length=8, direction='out')
plt.tick_params(labelsize=18)
#plt.tick_params(which='minor', left=True,bottom=True, colors='black', width=1,length=2, direction='out')
plt.xlabel('False Positive Rate',  fontsize=25)
plt.ylabel('True Positive Rate', fontsize=25)
#plt.title('Traing Set')
plt.legend(loc="lower right",  fontsize=18)# loc='best' loc="lower right"
#plt.show()
plt.savefig('attenrion-auc-train',dpi = 1200)
