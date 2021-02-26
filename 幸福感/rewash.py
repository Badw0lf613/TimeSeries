import pandas as pd
import csv
import numpy as np

data = pd.read_csv('happiness_train_complete_washed.csv',encoding='gbk')
k = data['happiness']
print(data.index[0])
for lin,q in zip(range(0,len(k)),k):
    if(q == -8):
        data = data.drop(index=[lin])
path = "happiness_train_complete_washed.csv"
data.to_csv(path,sep=",",index=False,header=True)
