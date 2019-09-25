# data_divider for 700-yeild-data by ratio
import pandas as pd

********************************************************

df = pd.read_csv('yield_702.csv')
df.drop_duplicates(keep='first', inplace=True) #去重

df = df.sample(frac = 1) #0-1 全部取用并打散
print(df.head(3))

#print(df.name)
df = df.sort_values(by='yield') # 升序按产率值排列

#print(df)
def get_index_by_yield(max_value):
    "按段值得到index，并返回"
    
    for i in range(len(df)):
        if float(df.iloc[i]['yield']) > float(max_value):
            #print(i)
            index = i
            break
    return index

"""def get_index(i,j):
    "i-起始值"
    "j-终止值"
    
    index = [get_index_by_yield(i), get_index_by_yield(j)]
    #index_name = "index" + "_" + str(i) + "_" + str(j)
    
    return index"""

"""index_0 = [0, get_index_by_yield(0)]
#print(index_0)

index_0_10 = get_index(0, 10)
index_10_20 = get_index(10, 20)
index_20_30 = get_index(20, 30)
index_30_40 = get_index(30, 40)
index_40_50 = get_index(40, 50)
index_50_60 = get_index(50, 60)
index_60_70 = get_index(60, 70)
index_70_80 = get_index(70, 80)
index_80_90 = get_index(80, 90)
index_90_100 = [get_index_by_yield(90), len(df)]"""

#print(index_90_100)
df_0 = df[0:get_index_by_yield(0)]
#print(df_1)
#for i in range(9):   
df_1 = df[get_index_by_yield(0): get_index_by_yield(10)]
df_2 = df[get_index_by_yield(10): get_index_by_yield(20)]
df_3 = df[get_index_by_yield(20): get_index_by_yield(30)]
df_4 = df[get_index_by_yield(30): get_index_by_yield(40)]
df_5 = df[get_index_by_yield(40): get_index_by_yield(50)]
df_6 = df[get_index_by_yield(50): get_index_by_yield(60)]
df_7 = df[get_index_by_yield(60): get_index_by_yield(70)]
df_8 = df[get_index_by_yield(70): get_index_by_yield(80)]
df_9 = df[get_index_by_yield(80): get_index_by_yield(90)]
df_10 = df[get_index_by_yield(90): len(df)]

DF = [df_0, df_1, df_2, df_3, df_4, df_5, df_6,df_7,df_8,df_9,df_10]
#print(DF)

Test = []
Train = []
for i in range(11):
    df = DF[i]
    cut_idx = int(round(0.2 * df.shape[0])) # 2:8
    df = df.sample(frac = 1)
    df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    Test.append(df_test)
    Train.append(df_train)

#print(len(Test))
# 连接test和train
df_test = Test[0]
for i in range(1, 11):
    df_test = pd.concat([df_test,Test[i]])
    
df_train = Train[0]
for i in range(1, 11):
    df_train = pd.concat([df_train,Train[i]])

#print(df_test)
df_test = df_test.sample(frac = 1)
df_train = df_train.sample(frac = 1)

df_test.to_csv(r"RAC_test_ratio.csv", index=None) 
df_train.to_csv(r"RAC_train_ratio.csv", index=None)

print(df_test.shape, df_train.shape)
