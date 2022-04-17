import pandas as pd

#Reading data
A = pd.read_csv("C:/Users/akaks/Downloads/50_Startups.csv")
A.head()
A.info()

# Checking Null Values
A.isnull().sum()                       #There are no null values
                                   
# Exploratory Data Aanlysis
#Here we will do EDA to check which features can be used to predict profit
A.corr()['PROFIT']                     #RND and MKT having the good Correlation with PROFIT

import seaborn as sb
sb.boxplot(A['STATE'],A['PROFIT'])     #from boxplot we can clearly see that there is no any solid relation between PROFIT and STATE, so we will ignore STATE.

# Defining X and Y

# from above EDA we saw that only RND and MKT have good correlation with PROFIT so we will be using those features as X to predict Profit(Y)
X = A[['RND']]
Y = A[['PROFIT']]

# Checking Skew
A.skew()                              #As skew may produce bias in model, but the selected features are normally distributed so there will be no problem

# Dividing data into training and testing set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=21)

# Creating Regression Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

# Training the Model
model = lm.fit(xtrain,ytrain)

b1 = round(model.coef_[0][0],2)
b0 = round(model.intercept_[0],2)
print("Y-intercept: ",b0,"\nslope: ",b1)

# Creating Predictions
pred_ts = model.predict(xtest)
pred_tr = model.predict(xtrain)

ytest['pred_profit'] = pred_ts
ytrain["pred_profit"] = pred_tr

from sklearn.metrics import mean_absolute_error
ts_err = mean_absolute_error(ytest['PROFIT'],ytest['pred_profit'])
tr_err = mean_absolute_error(ytrain['PROFIT'],ytrain['pred_profit'])
print("=========>>>>>TRAINING ERROR<<<<<<=======")
print(tr_err,'\n')
print("=========>>>>>TESTING ERROR<<<<<<========")
print(ts_err)

# Plotting regression results
import matplotlib.pyplot as plt
plt.scatter(X,Y,c='red')
plt.plot(X,b0+b1*X,c='blue')
plt.xlabel("RND")
plt.ylabel("PROFIT")
plt.title("Regression line for RND vs PROFIT")
