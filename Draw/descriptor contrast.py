import seaborn as sns
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline
#name :Name	ALogP	AMR	nAtom	nHeavyAtom	fragC	nHBAcc	nHBDon	nRing	nRotB	TopoPSA
df_train = pd.read_csv( 'train_data.csv', sep=',',encoding='gbk')
df_gen = pd.read_csv( 'generate_data.csv',sep=',',encoding='gbk')
df_div = pd.read_csv( 'generate_data.csv',sep=',',encoding='gbk')
# df=df1.join(df2.np)
# sns.distplot( df[df.cat == 'DIV'].np)
# sns.distplot( df[df.cat == 'NAT'].np)
# print(df_gen[:5])
# print(df_train.ALogP[:5])
# print(df_div.ALogP[:5])
# # col = df_gen.ALogP
# # arrs = col.values
# df_train.ALogP.dropna()
# print(1)
# df_gen.ALogP.dropna()
print(2)


sns.distplot(df_gen.TopoPSA[:100000])
sns.distplot(df_train.TopoPSA[:100000],color = 'r')
# sns.distplot(df_div.TopoPSA[:],color='c')
#sns.distplot(arrs)
# sum((df_gen.shape[0] - df_gen.count()) > 0)
plt.legend(('Training set','Generating set'),loc='best')
plt.xlabel('TopoPSA')  
plt.ylabel('KDE') 

sns.despine()
plt.savefig('Topo',dpi = 300)
