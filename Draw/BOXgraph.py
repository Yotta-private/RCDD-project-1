import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
tips = sns.load_dataset("tips")
# 绘制箱线图
ax = sns.boxplot(x=tips["total_bill"])
# 竖着放的箱线图，也就是将x换成y
ax = sns.boxplot(y=tips["total_bill"])
ax = sns.boxplot(x="day", y="total_bill", data=tips)
# ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
#                     data=tips, palette="Set3")
plt.show()

import pandas as pd
df = pd.read_csv('reprod.csv')
df.head(5)

df.iloc[:,1]

import seaborn as sns
import numpy as np
sns.set(style="ticks", palette="muted", color_codes=True)
plt.figure(figsize=(9, 6))
# sns.set_style("whitegrid")
# tips = sns.load_dataset("tips")
# 绘制箱线图
ax = sns.boxplot(x='Round',y='Reprod%', data = df,width = 0.6)
plt.xlabel('Round',fontsize=17)  
plt.ylabel('%Reprod',fontsize=17) 
plt.tick_params(labelsize=15)
plt.yticks(np.arange(22,31,2))
sns.despine()
plt.savefig('sca_sim',dpi = 220)
