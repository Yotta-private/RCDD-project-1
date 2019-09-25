import pandas as pd

*********************************************************

df = pd.read_csv('data.csv')
df.drop_duplicates(keep='first', inplace=True) #去重

df = df.sample(frac = 1) #0-1 全部取用并打散
cut_idx = int(round(0.3 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]

df_test.to_csv("test.csv", index=None) 
df_train.to_csv("train.csv", index=None)

print(df.shape, df_test.shape, df_train.shape)
