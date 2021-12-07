# JUMPlus Data Project 3: Iris Linear Regression
# Author: Tristan Kilper

# Requirements
print('  REQUIREMENTS')
print(' ')
# 1) Import libraries
print('1) Import required libraries')
print('--------------------------------------------------')
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
print('Python libraries pandas, sklearn, seaborn, and matplotlib imported to file!')
print(' ')

# 2) Read csv
print('2) Read csv')
print('--------------------------------------------------')
df = pd.read_csv('C:\Users\Tristan Kilper\Desktop\JUMP\JUMPlus\Project3Iris\JUMPlus_Iris\Iris.csv')
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
print(df[df['SepalWidth'] > 4])
print(' ')

# 5) Show a dataframe which has petalwidth greater than 1
print('5) Show a dataframe which has petalwidth greater than 1')
print('--------------------------------------------------')
print(df[df[PetalWidth] > 1])
print(' ')

# 6) Reterive records which have petalwidth more than 2
print('6) Reterive records which have petalwidth more than 2')
print('--------------------------------------------------')
print(df[df['PetalWength'] > 2])
print(' ')

# 7) Try to know the relationship between sepallength and petallength and draw a scatter plot between them and show the relationship between them
print('7) Try to know the relationship between sepallength and petallength and draw a scatter plot between them and show the relationship between them')
print('--------------------------------------------------')

print(' ')

# 8) Now apply species as hue in the same scatter plot for better visibility and understanding
print('8) Now apply species as hue in the same scatter plot for better visibility and understanding')
print('--------------------------------------------------')
print(' ')