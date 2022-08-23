#Importing required libraries.
from pyexpat import model
from statistics import linear_regression
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score

#Step 2 – Reading our input data for House Price Prediction
Data=pd.read_csv('USA_Housing.csv')
#print(Data.head())
#print('the shape of our Data USA_Housing {} '.format(Data.shape))

# Step 3 – Describing our data.
#print('the properties of our data \n {}'.format(Data.describe()))

#Step 4 – Analyzing information from our data.
#print(Data.info())

#Step 5 – visualize the interaction between the variables ouf our data 
#sns.pairplot(Data)
#plt.show()
'''form the plots shown we can infer that Price is highly correlated to Average Area Income''' 

#Step 6 –  split the data and scall it
X=Data.drop( ['Price','Address'],axis=1)
y=Data['Price']
'''print(X.head())
print(y.head())'''
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

#print(help(StandardScaler))
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

#step 7 - train the model using the linearRegression

model_1=LinearRegression()
model_1.fit(X_train_scaled,y_train)
pred=model_1.predict(X_test_scaled)
print('the fiting artion of the model_1 {}'.format(r2_score(y_test,pred)))
print('the accuracy of the model is {}'.format(model_1.score(X_test_scaled,y_test)))
print('the coeficent od the model are {}'.format(model_1.coef_))
print('the intercept of the model {}'.format(model_1.intercept_))

#step 8 - train the model using the Ridge
'''
param_grid={'alpha':[0.01,0.1,10,100,1000],}
grid_search_Ridge=GridSearchCV(Ridge(),param_grid,cv=5)
grid_search_Ridge.fit(X_train_scaled,y_train)
pred_Ridge=grid_search_Ridge.predict(X_test_scaled)
print('the best value for alpha is {}'.format(grid_search_Ridge.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search_Ridge.best_score_))
print('the fiting artion of the grid_search {}'.format(r2_score(y_test,pred_Ridge)))
print('the accuracy of the model is {}'.format(grid_search_Ridge.score(X_test_scaled,y_test)))
print("Best estimator:\n{}".format(grid_search_Ridge.best_estimator_))
'''
#print('the coeficent od the model are {}'.format(grid_search.coef_))
#print('the intercept of the model {}'.format(grid_search.intercept_))

#step 9 - train the model using the lasso

'''param_grid={'alpha':[0.01,0.1,10,100,1000],'max_iter':[10,100,1000,10000,100000,1000000]}
grid_search_Lasso=GridSearchCV(Lasso(),param_grid,cv=5)
grid_search_Lasso.fit(X_train_scaled,y_train)
pred_Ridge=grid_search_Lasso.predict(X_test_scaled)
print('the best value for alpha is {}'.format(grid_search_Lasso.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search_Lasso.best_score_))
print('the fiting artion of the grid_search {}'.format(r2_score(y_test,pred_Ridge)))
print('the accuracy of the model is {}'.format(grid_search_Lasso.score(X_test_scaled,y_test)))
print("Best estimator:\n{}".format(grid_search_Lasso.best_estimator_))
'''
''' since the tree models Lasso and Ridge and Linear Regression  performe nearly the same 
     we will stick with Linear Regression which is a more simple model '''

# Step 10 – Lets visualize our predictions of House Price Prediction.
sns.scatterplot(x=y_test,y=pred)
plt.show()

# Step 11 Plotting the residuals of our House Price Prediction model
sns.histplot((y_test-pred),bins=50,kde=True)
plt.show()