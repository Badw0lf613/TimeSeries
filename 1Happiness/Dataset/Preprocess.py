import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, RepeatedKFold
from scipy import sparse
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
from datetime import datetime

#导入数据
train_abbr=pd.read_csv("happiness_train_abbr.csv",encoding='ISO-8859-1')
train=pd.read_csv("happiness_train_complete.csv",encoding='ISO-8859-1')
test_abbr=pd.read_csv("happiness_test_abbr.csv",encoding='ISO-8859-1')
test=pd.read_csv("happiness_test_complete.csv",encoding='ISO-8859-1')
test_sub=pd.read_csv("happiness_submit.csv",encoding='ISO-8859-1')

#观察数据大小
print(test.shape)
print(test_sub.shape)
print(train.shape)

#简单查看数据
print(train.head())

#查看数据是否缺失
train.info(verbose=True,null_counts=True)
#查看label分布


y_train_=train["happiness"]
# y_train_.value_counts()
print(y_train_.value_counts())
#将-8换成3
y_train_=y_train_.map(lambda x:3 if x==-8 else x)
#让label从0开始
y_train_=y_train_.map(lambda x:x-1)
#train和test连在一起
data = pd.concat([train,test],axis=0,ignore_index=True)
#全部数据大小
print(data.shape)

#处理时间特征
data['survey_time'] = pd.to_datetime(data['survey_time'],format='%Y-%m-%d %H:%M:%S')
data["weekday"]=data["survey_time"].dt.weekday
data["year"]=data["survey_time"].dt.year
data["quarter"]=data["survey_time"].dt.quarter
data["hour"]=data["survey_time"].dt.hour
data["month"]=data["survey_time"].dt.month


# 把一天的时间分段
def hour_cut(x):
    if 0 <= x < 6:
        return 0
    elif 6 <= x < 8:
        return 1
    elif 8 <= x < 12:
        return 2
    elif 12 <= x < 14:
        return 3
    elif 14 <= x < 18:
        return 4
    elif 18 <= x < 21:
        return 5
    elif 21 <= x < 24:
        return 6

data["hour_cut"] = data["hour"].map(hour_cut)

#做问卷时候的年龄
data["survey_age"]=data["year"]-data["birth"]

#让label从0开始
data["happiness"]=data["happiness"].map(lambda x:x-1)

edu_other_data = data["edu_other"]

print(edu_other_data.value_counts())
print(edu_other_data.notnull())

#去掉三个缺失值很多的
data=data.drop(["edu_other"], axis=1)
data=data.drop(["happiness"], axis=1)
data=data.drop(["survey_time"], axis=1)

#是否入党
data["join_party"]=data["join_party"].map(lambda x:0 if pd.isnull(x)  else 1)