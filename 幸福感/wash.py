import pandas as pd
import csv
import numpy as np

csvfile = pd.read_csv('happiness_test_complete.csv',encoding='gbk')
# 前一列是有的，这一列大部分是空，去掉
df = csvfile.drop(['edu_other'],axis = 1)
# 有多少个18岁以下的孩子，负数没填代表没有，填0
df['minor_child'] = df['minor_child'].fillna(value=0)
list1 = []
# 对property独热编码，01看作二进制数，转整数
for a,b,c,d,e,f,g,h,i,j in zip(df['property_0'],df['property_1'],df['property_2'],df['property_3'],df['property_4'],df['property_5'],df['property_6'],df['property_7'],df['property_8'],df['property_other']):
    item = 0
    item = 1*a+2*b+4*c+8*d+16*e+32*f+64*g+128*h+256*i
    if('共' in str(j)):
        print(a,b,c,d,e,f,g,h,i,j)
        item = 1+2+4+8+16+32+64+128
    list1.append(item)
# 添加此新属性为property
df.insert(20,'property',list1)
# 删除原先的property
df = df.drop(['property_0','property_1','property_2','property_3','property_5','property_4','property_6','property_7','property_8','property_other'],1)
list1 = []
df['invest_other'] = df['invest_other'].fillna(value=-1)
# 对invest独热编码，01看作二进制数，转整数
for a,b,c,d,e,f,g,h,i,j in zip(df['invest_0'],df['invest_1'],df['invest_2'],df['invest_3'],df['invest_4'],df['invest_5'],df['invest_6'],df['invest_7'],df['invest_8'],df['invest_other']):
    item = 0
    item = 1*a+2*b+4*c+8*d+16*e+32*f+64*g+128*h+256*i
    if(not('-1' in str(j))):
        item = 1+2+4+8+16+32+64+128
    list1.append(item)
# 添加此新属性为invest
df.insert(71,'invest',list1)
# 删除原先的invest
df = df.drop(['invest_0','invest_1','invest_2','invest_3','invest_5','invest_4','invest_6','invest_7','invest_8','invest_other'],1)
list1 = df.keys()
# 根据index的说明，空数据视为不适用，填成-1
for item in list1:
    df[item] = df[item].fillna(value=-1)
path = "train\\happiness_test_complete_washed.csv"
df.to_csv(path,sep=",",index=False,header=True)
