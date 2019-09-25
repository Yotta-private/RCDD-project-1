from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns 
cm = plt.cm.get_cmap('RdYlBu')
sns.set(style="ticks", palette="muted", color_codes=True,font="Times New Roman")
# sns.set_style("whitegrid")  
# sns.set( style="ticks" , color_codes=True)  
plt.figure(figsize=(15, 12))
 
plt.scatter(df.iloc[1069:2235,0], df.iloc[1069:2235,1], marker='o',lw = 0,s = 70,
             color='blue', alpha=1, label='PCL')

p=plt.scatter(df.iloc[:1068,0], df.iloc[:1068,1], marker='o', lw = 0,s = 55,
            color='red', alpha=0.75, label='ZBL')

# plt.scatter(df.iloc[1069:2235,0], df.iloc[1069:2235,1], marker='o',lw = 0,s = 65,
#             color='r', alpha=0.75, label='Generating set')


plt.scatter(df.iloc[2236:3000,0], df.iloc[2236:3000,1], marker='o', s = 55,
             color='limegreen', alpha=0.9, label='ZLL')
# plt.colorbar(p) 
# plt.title('t-SNE of Phy ')
# plt.ylabel('variable X')
# plt.xlabel('Variable Y')
plt.legend(loc='best',fontsize=28)
# plt.xlim(-200,400) 
# plt.ylim(-75,110) 

plt.xlim(-0.05,0.8)
plt.ylim(0.15,1.1) 
plt.xticks(np.arange(0, 0.85,0.2))
plt.tick_params(labelsize='medium', width=5)

plt.tick_params(labelsize=22)
# plt.show()

plt.savefig('t-sme',dpi = 220)
