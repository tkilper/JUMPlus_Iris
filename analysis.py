# JUMPlus Data Project 3: Iris Linear Regression
# Author: Tristan Kilper

# Requirements
print(' ')
print('  REQUIREMENTS')
print(' ')
# 1) Import libraries
print('1) Import required libraries')
print('--------------------------------------------------')
import pandas as pd
from sklearn import linear_model, model_selection
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
print('Python libraries pandas, sklearn, seaborn, and matplotlib imported to file!')
print(' ')

# 2) Read csv
print('2) Read csv')
print('--------------------------------------------------')
df = pd.read_csv('C:/Users/Tristan Kilper/Desktop/JUMP/JUMPlus/Project3Iris/JUMPlus_Iris/Iris.csv')
print('csv read!')
print(' ')
"""
# 3) Show first records
print('3) Show first records')
print('--------------------------------------------------')
print(df.head(5))
print(' ')

# 4) Show a dataframe which has sepalwidth greater than 4
print('4) Show a dataframe which has sepalwidth greater than 4')
print('--------------------------------------------------')
print(df[df['SepalWidthCm'] > 4])
print(' ')

# 5) Show a dataframe which has petalwidth greater than 1
print('5) Show a dataframe which has petalwidth greater than 1')
print('--------------------------------------------------')
print(df[df['PetalWidthCm'] > 1])
print(' ')

# 6) Reterive records which have petalwidth more than 2
print('6) Reterive records which have petalwidth more than 2')
print('--------------------------------------------------')
print(df[df['PetalWidthCm'] > 2])
print(' ')

# 7) Try to know the relationship between sepallength and petallength and draw a scatter plot between them and show the relationship between them
print('7) Try to know the relationship between sepallength and petallength and draw a scatter plot between them and show the relationship between them')
print('--------------------------------------------------')
sns.scatterplot(data=df,x='SepalLengthCm',y='PetalLengthCm').set(title='SepalLength vs PetalLength (1)')
plt.show()
print('scat_slvspl1.png')
print(' ')
print('The relationship seems to be a direct relationship!')
print(' ')

# 8) Now apply species as hue in the same scatter plot for better visibility and understanding
print('8) Now apply species as hue in the same scatter plot for better visibility and understanding')
print('--------------------------------------------------')
sns.scatterplot(data=df,x='SepalLengthCm',y='PetalLengthCm',hue='Species').set(title='SepalLength vs PetalLength (2)')
plt.show()
print('scat_slvspl1.png')
print(' ')
"""

print('  MODELS')
print(' ')
print('1) MODEL 1')
# a) Create an object named as y which is storing the dataframe of a dependent variable names as 'sepallengthcm'
y = df[['SepalLengthCm']]
print(y)

# b) Create an object named as x which is storing the dataframe of an independent variable names as 'sepalwidthcm'
x = df[['SepalWidthCm']]
print(x)

# c) Divide the variables into x_train,x_test,y_train,y_test variables using train_test_split method carrying parameters named as x,y and test size should be 30%
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.3,random_state=42)

# d) Show first five records of all four variables / objects
print(x_train.head(5),x_test.head(5),y_train.head(5),y_test.head(5))

# e) Create an object named as lr and assign memory from linearregression() method.
lr = linear_model.LinearRegression()

# f) Fit both training set into fit method
lr.fit(x_train,y_train)

# g) Predict x_test from predict method and store the result into y_pred obect
y_pred = lr.predict(x_test) 
y_pred = pd.DataFrame(y_pred)
print(y_pred)

# h) Show first five records from actual and predicted objects
print(y_test.head(),y_pred.head())

# i) Try to find out mean_squared_error in prediction using method after passing parameter as y_test and y_pred ,mind the result
mse = mean_squared_error(y_test,y_pred)
print(f'The mean squared error is {mse}')
# The coefficients
print("Coefficients: \n", lr.coef_)
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# graph of the result
x_test.columns = ['x_test']
x_test = x_test.reset_index(drop=True)
y_test.columns = ['y_test']
y_test = y_test.reset_index(drop=True)
y_pred.columns = ['y_pred']
modeldata = pd.concat([x_test,y_test,y_pred], axis=1)
print(modeldata.head())
fig = plt.figure(figsize=(10,6))
sns.scatterplot(data=modeldata,x='x_test',y='y_test',color='blue').set(title='Linear Regression on Iris Sepal Width vs Sepal Length',xlabel="Sepal Width (cm)",ylabel="Sepal Length (cm)")
sns.lineplot(data=modeldata,x='x_test',y='y_pred',color='orange')
fig.legend(labels=['Model 1 Prediction','Original Data'])
plt.show()

print('2) MODEL 2')