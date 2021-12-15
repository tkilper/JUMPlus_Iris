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
from sklearn import linear_model, model_selection, preprocessing, cluster
from sklearn.metrics import mean_squared_error
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


print('  MODELS')
print(' ')
print('1) MODEL 1')
# a) Create an object named as y which is storing the dataframe of a dependent variable names as 'sepallengthcm'
y = df[['SepalLengthCm']]

# b) Create an object named as x which is storing the dataframe of an independent variable names as 'sepalwidthcm'
x = df[['SepalWidthCm']]

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

# h) Show first five records from actual and predicted objects
print(y_test.head(),y_pred.head())

# i) Try to find out mean_squared_error in prediction using method after passing parameter as y_test and y_pred ,mind the result
mse = mean_squared_error(y_test,y_pred)
print(f'The mean squared error is {mse}')

# graph of the result
x_test.columns = ['x_test']
x_test = x_test.reset_index(drop=True)
y_test.columns = ['y_test']
y_test = y_test.reset_index(drop=True)
y_pred.columns = ['y_pred']
modeldata = pd.concat([x_test,y_test,y_pred], axis=1)
fig = plt.figure(figsize=(10,6))
sns.scatterplot(data=modeldata,x='x_test',y='y_test',color='black').set(title='Linear Regression on Iris Sepal Width vs Sepal Length',xlabel="Sepal Width (cm)",ylabel="Sepal Length (cm)")
sns.lineplot(data=modeldata,x='x_test',y='y_pred',color='red')
fig.legend(labels=['Model 1 Prediction','Original Data'])
plt.show()


print('2) MODEL 2')

# a) Create an object named as y and store dataframe of sepallengthcm dependent variable
# Use from model 1
# b) Store 'sepalwidthcm','petallengthcm','petalwidthcm' dataframe in x as an independent variables
x2 = df[['SepalWidthCm','PetalLengthCm','PetalWidthCm']]

# c) Do train_test_split like you did in model 1 this time test_size is again 30%
x_train2, x_test2, y_train2, y_test2 = model_selection.train_test_split(x2,y,test_size=0.3,random_state=43)

# d) Fit both train set into fit method of linearregression
lr.fit(x_train2,y_train2)

# e) Predict x_test and store result into y_pred using predict method
y_pred2 = lr.predict(x_test2) 
y_pred2 = pd.DataFrame(y_pred2)

# f) Find out mean_squared_error of actual and predicted test set
mse = mean_squared_error(y_test2,y_pred2)
print(f'The mean squared error is {mse}')

# g) Describe which model is better and why?
print('Model 2 is better because there are more features than in model 1. More features in a linear regression analysis makes for a more accurate prediction, which is demonstrated in its lower mean squared error by a factor of 10.')


# BONUS: Logistic Regression
print('  BONUS: Logistic Regression')
print('1) MODEL 1')

# store logistic regression model in logr variable
logr = linear_model.LogisticRegression()

# find two centers using kmeans clustering to use as classifiers
centers = cluster.KMeans(n_clusters=2).fit(y)

# create the training and test datasets manually
x_trainlog, x_testlog, y_trainlog, y_testlog = model_selection.train_test_split(x,centers.labels_,test_size=0.3,random_state=42)

# train the model using the model 1 training datasets
logr.fit(x_trainlog,y_trainlog)

# calculate the logistic regression prediction using the test data
y_predlog = logr.predict(x_testlog)

# calculate the mean squared error
mse = mean_squared_error(y_testlog,y_predlog)
print(f'The mean squared error is {mse}')

# graph the result
x_testlog.columns = ['x_testlog']
x_testlog = x_testlog.reset_index(drop=True)
y_testlog = pd.DataFrame(y_testlog)
y_testlog.columns = ['y_testlog']
y_testlog = y_testlog.reset_index(drop=True)
for i in range(len(y_predlog)):
    if y_predlog[i] == 0:
        y_predlog[i] = centers.cluster_centers_[0]
    else:
        y_predlog[i] = centers.cluster_centers_[1]
y_predlog = pd.DataFrame(y_predlog)
y_predlog.columns = ['y_predlog']
modeldatalog = pd.concat([x_testlog,y_testlog,y_predlog], axis=1)
fig = plt.figure(figsize=(10,6))
sns.scatterplot(data=modeldata,x='x_test',y='y_test',color='black').set(title='Logistic and Linear Regression on Iris Sepal Width vs Sepal Length',xlabel="Sepal Width (cm)",ylabel="Sepal Length (cm)")
sns.lineplot(data=modeldatalog,x='x_testlog',y='y_predlog',color='blue')
sns.lineplot(data=modeldata,x='x_test',y='y_pred',color='red')
fig.legend(labels=['Linear Regression Prediction','Logistic Regression Prediction','Original Data'])
plt.show()

# Model 2
print('2) MODEL 2')

# create the training and test datasets
x_trainlog2, x_testlog2, y_trainlog2, y_testlog2 = model_selection.train_test_split(x2,centers.labels_,test_size=0.3,random_state=44)

# fit the training data to the logistic model
logr.fit(x_trainlog2,y_trainlog2)

# calculate the logistic regression prediction
y_predlog2 = logr.predict(x_test2)

# calculate the mean squared error
mse = mean_squared_error(y_test,y_predlog)
print(f'The mean squared error is {mse}')