#!/usr/bin/env python
# coding: utf-8

# # Read The Dataset

# In[1]:


import pandas as pd
A = pd.read_csv("C:/Users/akaks/Downloads/50_Startups.csv")


# In[2]:


A.head()


# In[3]:


A.info()


# # Checking Null Values

# In[4]:


A.isnull().sum()


# In[5]:


#There are no null values


# # Exploratory Data Aanlysis
Here we will do EDA to check which features can be used to predict profit
# In[6]:


A.corr()['PROFIT']


# In[7]:


#RND and MKT having the good Correlation with PROFIT


# In[8]:


import seaborn as sb
sb.boxplot(A['STATE'],A['PROFIT'])


# In[9]:


#from boxplot we can clearly see that there is no any solid relation between PROFIT and STATE, so we will ignore STATE.


# # Defining X and Y

# from above EDA we saw that only RND and MKT have good correlation with PROFIT so we will be using those features as X to predict Profit(Y)

# In[10]:


X = A[['RND']]
Y = A[['PROFIT']]


# # Checking Skew

# In[11]:


A.skew()


# In[12]:


#As skew may produce bias in model, but the selected features are normally distributed so there will be no problem


# # Dividing data into training and testing set

# In[13]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=21)


# # Creating Regression Model

# In[14]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# # Training the Model

# In[15]:


model = lm.fit(xtrain,ytrain)


# In[25]:


b1 = round(model.coef_[0][0],2)


# In[29]:


b0 = round(model.intercept_[0],2)


# In[30]:


print("Y-intercept: ",b0,"\nslope: ",b1)


# # Creating Predictions

# In[33]:


pred_ts = model.predict(xtest)


# In[41]:


pred_tr = model.predict(xtrain)


# In[35]:


ytest['pred_profit'] = pred_ts


# In[36]:


ytest


# In[43]:


ytrain["pred_profit"] = pred_tr


# In[45]:


#ytrain


# In[49]:


from sklearn.metrics import mean_absolute_error
ts_err = mean_absolute_error(ytest['PROFIT'],ytest['pred_profit'])
tr_err = mean_absolute_error(ytrain['PROFIT'],ytrain['pred_profit'])
print("=========>>>>>TRAINING ERROR<<<<<<=======")
print(tr_err,'\n')
print("=========>>>>>TESTING ERROR<<<<<<========")
print(ts_err)


# # Plotting regression results

# In[58]:


import matplotlib.pyplot as plt
plt.scatter(X,Y,c='red')
plt.plot(X,b0+b1*X,c='blue')
plt.xlabel("RND")
plt.ylabel("PROFIT")
plt.title("Regression line for RND vs PROFIT")


# In[ ]:




