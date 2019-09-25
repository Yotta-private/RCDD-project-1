"""#print(len(nums_a)) # 68
#print(len(nums_e)) # 50
Dict = dict(zip(data_name,data_yield))
data_read = []
for i in range(len(nums_e)):
    v = []
    name_e = nums_e[i]
    for j in nums_a:
        name_a = j
        try:
            value = Dict["RAC"+name_a+ name_e]
        except KeyError:
            value = 0.0000001 # 没做的
        v.append(value)
    data_read.append(v)
# print(len(data_read[0]))
data_draw = np.array(data_read)
data_draw = pd.DataFrame(data_draw, columns= np.array(nums_a), index= np.array(nums_e)) # 调整索引值
#print(data_draw)
#print(data_read)
#print(data_draw)"""

"""import seaborn as sns
sns.set()
# fig, ax = plt.subplots(figsize=(12,9))


fig= plt.figure(figsize=(21,15))
sns_plot = sns.heatmap(data_draw, annot=True, annot_kws={'size':9,'weight':'bold', 'color':'black'},
                       cmap="Blues", xticklabels=1, yticklabels=1, center=20, mask=(data_draw==0.0000001),
                      linewidths = 0.01, linecolor= "white") # 8表示步长
# tick_params 中 direction='in'表示刻度线位于内侧，另外还有参数 out,inout # heatmap 刻度字体大小
sns_plot.tick_params(labelsize=10, direction='in')
# colorbar 刻度线设置
cax = plt.gcf().axes[-1]
# colorbar 刻度字体大小
# colorbar 中 top='off', bottom='off', left='off', right='off'表示上下左右侧的刻度线全部不显示
cax.tick_params(labelsize=10, direction='in', top=False, bottom=False, left=False, right=False)
# fig.savefig("heatmap.pdf", bbox_inches='tight') # 减少边缘空白

#mask对某些矩阵块的显示进行覆盖

plt.show()"""

data = pd.read_csv("RAC_data.csv")
data_yield = list(data["yield"])
data_yield.sort()
data = data_yield
#print(data)
data_cut0 = [x for x in data if x > 0]
#print(data_cut0)
#print(len(data_cut0))
count = []
for i in range(10):
    d =  [x for x in data_cut0 if x >= i *10 if x < 10*(i+1)]
    count.append(len(d))
# [405, 94, 56, 36, 22, 28, 20, 15, 5, 15, 6]
"""count_0 = len(data) - len(data_cut0 )
#print(count_0 )
data_count = [10 for _ in range(94) ] + [20 for _ in range(56)] 
data_count +=  [30 for _ in range(36)] + [40 for _ in range(22)] 
data_count+= [50 for _ in range(28)] + [60 for _ in range(20)] + [70 for _ in range(15)] + [80 for _ in range(5)] + [90 for _ in range(15)]
data_count+=[100 for _ in range(6)]
print(data_count)"""

#sns.set(style="ticks",font="Times New Roman")
sns.color_palette("hls", 10)
plt.style.use('default')
#sns.set_style('darkgrid')
#sns.set_context('paper')

plt.figure(figsize=(8, 6))
#sns.distplot(data_cut0, color = "gray", rug_kws = {'color':'gray'},
            #hist_kws = {'histtype':'stepfilled', 'linewidth':0.01, 'alpha':1, 'color':'darkgray'})
#plt.legend(fontsize=8)
sns.distplot(data_cut0,  hist_kws = {'histtype':'stepfilled', 'linewidth':0.01, 'alpha':1, 'color':'dimgray'},
             color='gray',label = 'RAC set')
#plt.xlabel("yield", fontsize=10)  
#plt.ylabel('distribution',fontsize=10) 
#plt.tick_params(labelsize='medium', width=1)
#plt.tick_params(labelsize=8)
#plt.xticks(np.arange(0, 100,10))
#plt.yticks(np.arange(0, 0.1,0.1))
#sns.despine()
#plt.savefig('yield-distribution',dpi = 800)

#plt.legend(('RAC set'),loc='best',fontsize=17)
plt.xlabel('yield',fontsize=10)  
plt.ylabel('Distribution',fontsize=10) 
plt.tick_params(labelsize='medium', width=1)
plt.xticks(np.arange(0, 100,10))
#plt.yticks(np.arange(0, 0.1,0.1))
sns.despine()
plt.savefig('yield-distribution-3',dpi = 800)
