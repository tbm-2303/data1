#!/usr/bin/env python
# coding: utf-8

# # Mini-Project 3 

# ## 1. Introduction

# ### Case overview
# We are given a dataset containing data from the property market in King
# County, USA.
# The task is to use the dataset for training a regression model that can be used for
# prediction of prices of properties not listed in the file.
# 
# Our response variable is price, but we have 20 potential explanatory variables. so we are going to need to figure our which features are best used as explanatory variables.
# 
# 
# ### The Dataset
# ##### Here is a link to the dataset: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?datasetId=128&sortBy=voteCount
# 
# ### Overview of the variables
# 
# #### There are 21 variables in this data set:
# 
# - id -  House ID
# - date -  Date house sold
# - price -  House price
# - bedrooms -  Number of bedrooms
# - bathrooms -  Number of bathrooms
# - sqft_living -  Living room size (in sqft)
# - sqft_lot -  Lot size (in sqft)
# - floors -  Number of floors
# - waterfront -  Has access to waterway (0 = no; 1 = yes)
# - view -  View
# - condition -  House condition (1 = bad; 5 = perfect)
# - grade -  House grade
# - sqft_above -  Above size (in sqft)
# - sqft_basement -  Basement size (in sqft)
# - yr_built -  Year when house was built
# - yr_renovated -  Year when house was renovated
# - zipcode -  House zipcode
# - lat -  House latitude
# - long -  House longitude
# - sqft_living15 -  Living15 (in sqft)
# - sqft_lot15 -  Lot15 (in sqft)
# 
# 

# ## 2. Import libraries

# In[7]:


import numpy as np #linear algebra
import pandas as pd # data processing 
import seaborn as sns #plot data 
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sm
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#show all columns at once 
pd.set_option('display.max_columns', None)


# ##  3. Read the data into a pandas dataframe

# In[8]:


data = pd.read_csv("/Users/cej12/Skrivebord/Untitled Folder/data/house-data.csv")
data


# ## 4. Initial exploration
# When you get a brand new dataset, it can be very hard to know what to look for. Where should you start your exploration. Well, before you start the actually exploring, you should aim to familiarize yourself with the dataset. Here are some common functions to help us get familiarized with a dataset. 
# - data.head()
# - data.columns
# - data.shape
# - data.info()
# - data.isnull().sum()
# - data.describe()
# - data.describe().transpose()
# - data
# - list(data)
# - data.set_option('display.float_format', lambda x: '%.3f' % x)
# - data.sample()
# 
# 
# Once we have an idea for the data we can start to look for potential problems. Are there null values? Are there any fractions? Which variables are continuous and which are categorical. Do we need to binarize som categorical string values into numerical values?. Is the data for some variables heavily skewed or doesnt make sence(why is there 13 houses with 0 bedrooms). Are there any insane outliers?

# ### 4.1. visualisation

# In[200]:


plt.figure(figsize=(12,10))
sns.histplot(data.price, kde=True)


# In[201]:


sns.displot(data.price,kde=True)


# In[202]:


sns.countplot(data, x='bedrooms')


# ## 5. Investigate correlation with features
# We can use our intuition to select some features that we expect have high cooralation. Common sence would suggest that theese features would have a somewhat noticable effect on property prices: 'sqft_living', 'bedrooms', 'bathrooms', 'grade', 'waterfront', 'view' and 'sqft_lot'. 
# 
# - With 'bedrooms', 'bathrooms', 'waterfront', 'view' and 'grade' I preferred boxplot because we have numerical data but they are not continuous. 
# 
# - For 'sqft_living', 'sqft_lot' i used scatterplot since they have continuous data.
# 

# In[203]:


plt.figure(figsize=(10,5))
sns.scatterplot(x='sqft_living', y='price', data=data)


# In[204]:


plt.figure(figsize=(10,5))
sns.scatterplot(x='sqft_lot', y='price', data=data)


# In[205]:


plt.figure(figsize=(10,6))
sns.boxplot(data=data, x='bedrooms', y='price')


# In[206]:


plt.figure(figsize=(10,6))
sns.boxplot(data=data,x='bathrooms',y='price')


# In[207]:


sns.boxplot(data=data, x='grade', y='price')


# In[208]:


sns.boxplot(data=data, x='waterfront', y='price')


# In[209]:


sns.boxplot(data=data, x='view', y='price')


# ### 5.1 Check  coorelation 
# in the table below we see that our intuition was somewhat correct The 5 strongest coorelated features with price are:
# - 1. sqft_living
# - 2. grade
# - 3. sqft_above (since both sqft_above and living both are highly related with price sqft_above should probably be removed if sqft_above and sqft_living have high inter-dependency)
# 
# - 4. sqft_living15
# - 5. bathrooms
# 
# waterfront seems to be worse then we anticipated from the boxplots. It's important to remember that the dataset is far from perfect. Some people would remove some of the big outliers to improve the model, but in the real world there will always be statistical outliers and randomness. We might make another version where we remove the outliers to see if it improves the model. But this end up not effecting the accuracy of the model. The model aims to predict the real world. and a dataset made from the real world will almost curtainly contain outliers. So if we can keep theese properties eventhough they dont follow the overall trends in the property market, the model might be better to deal with such outliers. 
# 

# In[237]:


data.corr(numeric_only=True)['price'].sort_values()


# ### 5.2 Analysing the results 
# It seems that there is not a perfect linear relationship between the price and these features. When we look at the above boxplots, grade and waterfront effect price visibly. On the other hand, view seem to effect less but it also has an effect on price. sqft_living seems to have some cooralation but sqft_lot seems to have a significantly weaker cooralation. Now we have some idea of which features effect price we can proceed. 
# 
# When we model a linear relationship between a response and just one explanatory variable, this is called simple linear regression. I want to predict house prices and then, our response variable is price. Our strongest feature seens to be sqft_living. But given our investegation even our strongest feature seems like it would'nt allow for accurate predictions from a simple linear regression model. We are still going to try. But we problably need to include some more features to impove the model. But having too many features in a model is not always a good thing because it might cause overfitting and worser results when we want to predict values for a new dataset. Thus, if a feature does not improve your model a lot, not adding it may be a better choice.
# 
# Before we decide on additional features an important thing is correlation. If there is very high correlation between two features, keeping both of them is not a good idea most of the time not to cause overfitting. For instance, if there is overfitting, we may remove sqt_above or sqt_living because they are highly correlated. This relation can be estimated when we look at the definitions in the dataset but to be sure correlation matrix should be checked. However, this does not mean that you must remove one of the highly correlated features. For example: bathrooms and sqft_living. They are highly correlated but I do not think that the relation among them is the same as the relation between sqft_living and sqft_above. 
# 
# lets take at look at a correlation matrix:

# ### 5.3 correlation matrix
# fsdfdsfsdfsdfsdfsfsddsf
# 
# dfsdfdsfsd
# 
# fsdfsdf
# 
# sdfsdf

# In[6]:


#data = data.drop('date',axis=1) (need to remove 'date' before calling data.corr() since date contains non-intable data) 
corr_matrix = tmp2.corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, annot=True)


# ## 6 Cleaning 
# remove useless columns:
# - id
# 
# - date
# 
# other things to think about:
# - should we remove the most expensive properties
# - should we remove properties with 0 bedrooms
# - should we remove properties with 7-8 or more bedrooms 
# - should take into account -> geography, demographics, age of property, time since renovation etc. 

# In[213]:


#data = data.drop('id', axis = 1)
#data = data.drop('date',axis=1)
#data = data.drop('zipcode', axis = 1)


# # 7 Train a Model

# ### 7.1 Splitting Dataset

# In[18]:


tmp = data.drop(['id'], axis = 1)
tmp2 = tmp.drop(['date'], axis=1)
tmp2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


target = tmp2['price']


# In[14]:


tmp3 = tmp2.drop(['price'], axis = 1)
tmp3


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(tmp3, target, random_state=42, train_size=0.8, shuffle=True)
print ("train size={}, test_size={}, total_size={}".format(X_train.shape[0], X_test.shape[0], data.shape[0]))


# In[17]:


model = LinearRegression()
model.fit(X_train, y_train)
print("num_ftrs = {}, num_coeff = {} ".format(X_train.shape[1], len(model.coef_)))
reg_coeff = dict(zip(tmp3.columns, model.coef_))
print(reg_coeff)


# In[272]:


y_pred_train = model.predict(X_train)
print("Quality Test {}".format(mean_squared_error(y_train, y_pred_train)))
y_pred = model.predict(X_test)
print("Quality Control {}".format(mean_squared_error(y_test, y_pred)))


# In[273]:


sns.barplot(x = X_train.columns, y=model.coef_)
plt.xticks(rotation=90);


# In[267]:


print("Linear regression r2 score: ", r2_score(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[234]:


data2 = data.drop(['id'], axis = 1)
target = data2['price']
data2 = data.drop(['price'], axis = 1)
data2 = data.drop("date",axis=1)
data2


# In[228]:


X_train, X_test, y_train, y_test = train_test_split(data2, target, random_state=42, train_size=0.8, shuffle=True)

print ("train size={}, test_size={}, total_size={}".format(X_train.shape[0], X_test.shape[0], data.shape[0]))


# In[229]:


model = LinearRegression()
model.fit(X_train, y_train)
print("num_ftrs = {}, num_coeff = {} ".format(X_train.shape[1], len(model.coef_)))
reg_coeff = dict(zip(data2.columns, model.coef_))
print(reg_coeff)


# In[197]:


y_pred_train = model.predict(X_train)
print("Quality Test {}".format(mean_squared_error(y_train, y_pred_train)))
y_pred = model.predict(X_test)
print("Quality Control {}".format(mean_squared_error(y_test, y_pred)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[126]:





# In[127]:


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=123, test_size=0.15) 


# In[135]:


# creating an instance of Linear Regression model
myreg = LinearRegression()


# In[138]:


# fit it to our data
myreg.fit(X_train, y_train)


# In[139]:


# get the calculated coefficients
a = myreg.coef_
b = myreg.intercept_


# In[143]:


a


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42, train_size=0.8, shuffle=True)
print ("train size={}, test_size={}, total_size={}".format(X_train.shape[0], X_test.shape[0], data.shape[0]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[86]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=data,x='price',y='long')


# In[87]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=data,x='price',y='lat')


# In[88]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=data,x='long',y='lat',hue='price')


# In[89]:


data.sort_values('price',ascending=False).head(20)


# In[90]:


len(data)*0.01


# In[91]:


perc_99 = data.sort_values('price', ascending=False).iloc[216:]


# In[92]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=perc_99, x='long', y='lat', hue='price', alpha=0.2, palette = 'YlOrBr', edgecolor=None)


# In[95]:


sns.color_palette("YlOrBr", as_cmap=True)


# ## 5. cleaning the data
# 
# 

# In[ ]:




