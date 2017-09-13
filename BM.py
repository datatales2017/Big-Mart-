# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 16:12:04 2017

@author: manasa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Train.csv')
datatest = pd.read_csv('Test.csv')
test = pd.read_csv('Test.csv')

data.head()
data.describe()

#Finding the null values
nulls = data.apply(lambda x:sum(x.isnull()))
testnulls = datatest.apply(lambda x:sum(x.isnull()))

data['Item_Fat_Content'].unique()

#Here Low Fat,low fat & LF can be combinned under one category.same with Reg and Regular

data['Item_Fat_Content'].replace(({'reg':'Regular'}),inplace=True)
data['Item_Fat_Content'].replace(({'low fat':'Low Fat'}),inplace=True)
data['Item_Fat_Content'].replace(({'LF':'Low Fat'}),inplace=True)

data['Item_Fat_Content'].unique()

data.groupby(['Item_Fat_Content'])['Item_Weight'].mean()

#Item_Fat_Content
#Low Fat    12.937387
#Regular    12.711654

data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True);

data['Outlet_Size'].fillna("NA",inplace=True)
data['Outlet_Size'].unique()

data.describe()

plt.plot(data['Item_Weight'],data['Item_Outlet_Sales'])
plt.xlabel("Item Weight")
plt.ylabel("Item Outlet Sales")
plt.title("Item Weight v/s Item Outlet Sales")
plt.show()

sns.jointplot('Item_Weight','Item_Outlet_Sales',data=data)

sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',data=data,hue='Item_Fat_Content')

data['Item_Fat_Content'].unique()
data['Item_Type'].unique()
data['Outlet_Type'].unique()
data['Outlet_Identifier'].unique()
data['Outlet_Size'].unique()
data['Outlet_Location_Type'].unique()
data['Outlet_Type'].unique()


plt.figure(figsize=(25,15))
sns.boxplot(x='Item_Type',y='Item_Outlet_Sales',data=data,hue='Item_Fat_Content')
plt.xlabel('Item Types')
plt.ylabel("Outlet Sales")

data.head(25)
#Calculating the numbers of years the store is operating
data['Outlet_Establishment_Year'].max()
data['NoOfYears'] = data['Outlet_Establishment_Year'].max() - data['Outlet_Establishment_Year']
data['NoOfYears'].describe()

#Dropping the Outlet_Establishment_Year feature
data = data.drop("Outlet_Establishment_Year",axis=1)

data['Item_Ids']=data['Item_Identifier'].str[:2]
data['Item_Ids'].unique()

#Dropping item identifier
data = data.drop("Item_Identifier",axis=1)

#Creating dummy variables for all categorical features
data.info()
y = data['Item_Outlet_Sales']
X = data[['Item_Weight', 'Item_Fat_Content','Item_Visibility','Item_Type',
          'Item_MRP','Outlet_Identifier','Outlet_Size', 'Outlet_Location_Type',
          'Outlet_Type','NoOfYears','Item_Ids']]
X = pd.get_dummies(X)          

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor
reg1 = RandomForestRegressor(n_estimators=2000,criterion='mse',max_features = 'sqrt',
                             max_depth=5,min_samples_split=5,random_state=42)
                             
reg1.fit(X_train,y_train)
ypred = reg1.predict(X_test)

ypred.min()
ypred.max()
plt.figure(figsize=(10,8))
sns.regplot(ypred,y_test,scatter=True)
plt.xlabel("Predicted Outlet Sales")
plt.ylabel("Original Outlet Sales")
plt.title("OriginalOutlet Sales v/s Predicted Outlet Sales")

#############################TEST DATA##################################
datatest['Item_Fat_Content'].unique()

#Here Low Fat,low fat & LF can be combinned under one category.same with Reg and Regular

datatest['Item_Fat_Content'].replace(({'reg':'Regular'}),inplace=True)
datatest['Item_Fat_Content'].replace(({'low fat':'Low Fat'}),inplace=True)
datatest['Item_Fat_Content'].replace(({'LF':'Low Fat'}),inplace=True)

datatest['Item_Fat_Content'].unique()

datatest.groupby(['Item_Fat_Content'])['Item_Weight'].mean()

#Item_Fat_Content
#Low Fat    12.860395
#Regular    12.394528

datatest['Item_Weight'].fillna(datatest['Item_Weight'].mean(),inplace=True);

datatest['Outlet_Size'].fillna("NA",inplace=True)
datatest['Outlet_Size'].unique()

datatest.describe()

plt.plot(datatest['Item_Weight'],datatest['Item_Outlet_Sales'])
plt.xlabel("Item Weight")
plt.ylabel("Item Outlet Sales")
plt.title("Item Weight v/s Item Outlet Sales")
plt.show()

sns.jointplot('Item_Weight','Item_Outlet_Sales',data=datatest)

sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',data=datatest,hue='Item_Fat_Content')

datatest['Item_Fat_Content'].unique()
datatest['Item_Type'].unique()
datatest['Outlet_Type'].unique()
datatest['Outlet_Identifier'].unique()
datatest['Outlet_Size'].unique()
datatest['Outlet_Location_Type'].unique()
datatest['Outlet_Type'].unique()


plt.figure(figsize=(25,15))
sns.boxplot(x='Item_Type',y='Item_Outlet_Sales',datatest=datatest,hue='Item_Fat_Content')
plt.xlabel('Item Types')
plt.ylabel("Outlet Sales")

datatest.head(25)
#Calculating the numbers of years the store is operating
datatest['Outlet_Establishment_Year'].max()
datatest['NoOfYears'] = datatest['Outlet_Establishment_Year'].max() - datatest['Outlet_Establishment_Year']
datatest['NoOfYears'].describe()

#Dropping the Outlet_Establishment_Year feature
datatest = datatest.drop("Outlet_Establishment_Year",axis=1)

datatest['Item_Ids']=datatest['Item_Identifier'].str[:2]
datatest['Item_Ids'].unique()

#Dropping item identifier
datatest = datatest.drop("Item_Identifier",axis=1)       
     
datatest = pd.get_dummies(datatest)
SalesPred = reg1.predict(datatest)
                           
submission = pd.DataFrame()
submission['Item_Identifier']=test['Item_Identifier']
submission['Outlet_Identifier']=test['Outlet_Identifier']
submission['Item_Outlet_Sales']=SalesPred
submission.to_csv('M:/AV/bigmart/submission.csv',index=False)

from xgboost import XGBRegressor
reg2 = XGBRegressor(max_depth=5, learning_rate=0.5, n_estimators=3000, silent=True, objective="reg:linear", 
                    nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, 
                    colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
                    scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
reg2.fit(X_train,y_train)
SalesPred_xgb = reg2.predict(datatest)

submission_xgb = pd.DataFrame()
submission_xgb['Item_Identifier']=test['Item_Identifier']
submission_xgb['Outlet_Identifier']=test['Outlet_Identifier']
submission_xgb['Item_Outlet_Sales']=SalesPred_xgb
submission_xgb.to_csv('M:/AV/bigmart/submission_xgb.csv',index=False)
