import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  #字体管理器

plt.xticks(rotation=270)
data = pd.read_csv('happiness_train_complete_washed.csv',encoding='gbk')
path = 'happiness_train_complete_washed.csv'
name = data.columns.values.tolist()
name.remove("happiness")
happ = pd.Series(data['happiness'])
corr = []
for i in name:
    q = data[i]
    s1 = pd.Series(q)
    try:
        corr.append(happ.corr(s1))
    except TypeError:
        corr.append(0)

ln1= plt.plot(name,corr,color='red',linewidth=2.0,linestyle='-')
plt.title("Correlation") #设置标题及字体
ax = plt.gca()
ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
ax.spines['top'].set_color('none')    # top边框属性设置为none 不显示
plt.show()
'''
data = pd.read_csv('happiness_train_complete_washed.csv',encoding='gbk')
data2 = pd.read_csv('happiness_test_complete_washed.csv',encoding='gbk')
path = 'happiness_train_complete_washed.csv'
name = data.columns.values.tolist()
name.remove("happiness")
happ = pd.Series(data['happiness'])
for i in name:
    q = data[i]
    s1 = pd.Series(q)
    try:
        corr = happ.corr(s1)
        if corr < 0.2:
            data = data.drop([i],axis=1)
            data2 = data2.drop([i],axis=1)
    except TypeError:
        continue
data.to_csv("happiness_train_complete_washed.csv",index=False,header=True)
data2.to_csv("happiness_test_complete_washed.csv",index=False,header=True)
''' 
