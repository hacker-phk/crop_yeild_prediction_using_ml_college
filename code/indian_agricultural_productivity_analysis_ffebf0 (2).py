# -*- coding: utf-8 -*-
"""indian-agricultural-productivity-analysis-ffebf0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-e-tONCofDhgBPGy1t5nPZj605ANR7Q6

# Indian Agricultural Crop Yield Predictions using  Machine Learning  Algorithms

## Importing Libraries
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

from google.colab import files
u=files.upload()

# Loading the dataset
df = pd.read_csv('crop_yield.csv')
df.head()

df.tail()

print("Shape of the dataset : ",df.shape)

"""# Preprocessing of the dataset"""

df.isnull().sum()

df.info()

# to check the unique values
for i in df.columns:
    print("******************************",i,"*********************************")
    print()
    print(set(df[i].tolist()))
    print()

# Check the duplicates record
df.duplicated().sum()

df.describe()

"""# Visualization"""

sns.scatterplot(x = df['Annual_Rainfall'], y = df['Yield'])
plt.show

"""# Year wise analysis of agricultural production"""

df_year = df[df['Year']!=2020]

year_yield = df_year.groupby('Year').sum()
year_yield

plt.figure(figsize = (12,5))
plt.plot(year_yield.index, year_yield['Yield'],color='blue', linestyle='dashed', marker='o',
        markersize=12, markerfacecolor='yellow')
plt.xlabel('Year')
plt.ylabel('Yield')
plt.title('Measure of Yield over the year')
plt.show()

"""#### It can be observed that the yield has increased over the year, but after 2014 it is showing the declining trend. Reasons can be climate change, decrease in soil fertility"""

plt.figure(figsize = (12,3))
plt.plot(year_yield.index, year_yield['Area'],color='blue', linestyle='dashed', marker='o',
        markersize=12, markerfacecolor='red')
plt.xlabel('Year')
plt.ylabel('Area')
plt.title('Area under cultivation over the year')
plt.show()

"""### It can be observed that the area under cultivation has increased substantially. Either with the help of fertilizer and more irrigation fallow land is now under cultivation or area under forest is used for agriculture."""

plt.figure(figsize = (12,3))
plt.plot(year_yield.index, year_yield['Fertilizer'],color='blue', linestyle='dashed', marker='o',
        markersize=12, markerfacecolor='green')
plt.xlabel('Year')
plt.ylabel('Fertilizer')
plt.title('Use of Fertilizer over the year')
plt.show()

"""### The use of Fertilizer in the fields is increasing"""

plt.figure(figsize = (12,3))
plt.plot(year_yield.index, year_yield['Pesticide'],color='red', linestyle='dashed', marker='o',
        markersize=12, markerfacecolor='cyan')
plt.xlabel('Year')
plt.ylabel('Pesticide')
plt.title('Use of Pesticide over the Year')
plt.show()

"""# State wise analysis of agricultural production"""

df_state = df.groupby('State').sum()
df_state.sort_values(by = 'Yield', inplace=True, ascending = False)
df_state

df_state['Region'] = ['States' for i in range(len(df_state))]

fig = px.bar(df_state, x='Region', y = 'Yield', color=df_state.index, hover_data=['Yield'])
fig.show()

### From the above graph it can be observed that the yield of West Bengal is highest. Reason can be more annual rainfall, use of fertilizers

"""plt.figure(figsize = (15,8))
sns.barplot(x = df_state.index, y=df_state['Annual_Rainfall'], palette = 'gnuplot')
plt.xticks(rotation = 45)
plt.show()
"""

plt.figure(figsize=(12,5))
sns.scatterplot(x=df_state.index, y=df_state['Fertilizer'], palette='spring', hue = df_state['Yield'])
plt.xticks(rotation=45)
plt.title('Use of Fertilizer in Different States')
plt.show()

"""### Observations:
* Annual Rainfall is highest in Chattisgarh but the yield is not the highest.
* West Bengal has the maximum yield
* Uttar Pradesh, Haryana, Maharashtra are using high amount of fertilizer but yield is not high reason can be low annual rainfall

# Season wise analysis
"""

df_Seas = df[df['Season']!='Whole Year ']

df_season = df_Seas.groupby('Season').sum()
df_season

fig = px.bar(df_season, y = 'Area', color=df_season.index, hover_data=['Area'],text = 'Area')
fig.show()

fig = px.sunburst(df_season, path=[df_season.index, 'Yield'], values='Yield',
                  color=df_season.index, hover_data=['Yield'])
fig.show()

"""## Observations:
* Area under cultivation in Kharif season is highest, second is Rabi season
* Crops in autumn, summer are not grown over large area
* Yield in India is maximum in Kharif season

# Crop wise Analysis
"""

# Where the Yield is zero
df_yz = df[df['Yield']==0]
df_yz.shape

df_yz.head()

plt.figure(figsize = (25,15))
sns.catplot(y="State", x="Crop",data=df_yz, aspect = 3, palette ='inferno')
plt.xticks(rotation=45)
plt.title('States and the Crops where yield is zero')
plt.show()

df_ynz = df[df['Yield']>0]  # where yield is more than zero
df_crop = df_ynz.groupby('Crop').sum()
df_crop

plt.figure(figsize = (25,8))
plt.plot(df_crop.index, df_crop['Fertilizer'],color='red', linestyle='dashed', marker='o',
        markersize=12, markerfacecolor='cyan')
plt.xlabel('Crops')
plt.ylabel('Fertilizer')
plt.title(' Use of Fertilizer in different Crops')
plt.xticks(rotation=30)
plt.show()

"""### The amount of Fertilizer used is maximum in Rice Crop
### The second crop to use more fertilizer is Wheat crop
"""

plt.figure(figsize = (25,8))
plt.plot(df_crop.index, df_crop['Area'],color='indigo', linestyle='dashed', marker='o',
        markersize=12, markerfacecolor='fuchsia')
plt.xlabel('Crops')
plt.ylabel('Area under cultivation')
plt.xticks(rotation=30)
plt.show()

"""#### Area under cultivation is larger for Rice and Wheat crops

# Analysis of Wheat crop
"""

df_wheat = df[df['Crop']=='Wheat']
df_wheat.reset_index(drop=True,inplace=True)
df_wheat

df_wheat1 = df_wheat[df_wheat['Year']!=2020]
df_wheat_year = df_wheat1.groupby('Year').sum()
df_wheat_year

plt.figure(figsize = (12,5))
plt.plot(df_wheat_year.index, df_wheat_year['Yield'],color='red', linestyle='dashed', marker='o',
        markersize=12, markerfacecolor='blue')
plt.xlabel('Year')
plt.ylabel('Yield')
plt.title('Yield of Wheat Crop over the Years')
plt.show()

"""### Checking the co-relation in the dataset using heatmap

## Fertilizer and Pesticide are showing the same corelation. Hence, have to drop one column to avoid Multicollinearity

# Modelling
"""

df1 = df.copy()
df1 = df1.drop(['Year','Pesticide'], axis = 1)

# To check the distribution of dataset
plt.figure(figsize=(15,20))
plt.subplot(4,2,1)
sns.distplot(df1['Area'],bins = 20,color = 'red')
plt.subplot(4,2,2)
sns.distplot(df1['Production'],bins = 10,color = 'green')
plt.subplot(4,2,3)
sns.distplot(df1['Annual_Rainfall'],bins = 10,color = 'blue')
plt.subplot(4,2,4)
sns.distplot(df1['Fertilizer'],bins = 10, color = 'black')
plt.show()

# Q-Q plot of the dataset
import scipy.stats as stats

plt.figure(figsize=(15,20))
plt.subplot(4,2,1)
stats.probplot(df1['Area'], dist = 'norm', plot = plt)
plt.subplot(4,2,2)
stats.probplot(df1['Production'], dist = 'norm', plot = plt)
plt.subplot(4,2,3)
stats.probplot(df1['Annual_Rainfall'], dist = 'norm', plot = plt)
plt.subplot(4,2,4)
stats.probplot(df1['Fertilizer'], dist = 'norm', plot = plt)
plt.show()

"""### Data distribution have right skewness - to remove skewness using transformation approach
The algorithm is more likely to be biased when the data distribution is skewed

# One-Hot Encoding
"""

category_columns = df1.select_dtypes(include = ['object']).columns
category_columns

df1 = pd.get_dummies(df1, columns = category_columns, drop_first=True)

df1.shape

df1.head()

"""### Split the data into dependent and independent variable"""

x = df1.drop(['Yield'], axis = 1)
y = df1[['Yield']]

print(x.shape)
y.shape

x.head()

y.head()

"""### Splitting  the data set into train and test set"""

#split the data into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

x_train.shape, x_test.shape, y_train.shape,y_test.shape

"""# Power Transformation using the method 'Yeo-Johnson'"""

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')

x_train_transform1 = pt.fit_transform(x_train)
x_test_transform1 = pt.fit_transform(x_test)

df_trans = pd.DataFrame(x_train_transform1, columns=x_train.columns)
df_trans.head()

"""## After Transformation, there is no need for Standardization of the data"""

plt.figure(figsize=(15,20))
plt.subplot(4,2,1)
sns.distplot(df_trans['Area'],bins = 20,color = 'red')
plt.subplot(4,2,2)
sns.distplot(df_trans['Production'],bins = 10,color = 'green')
plt.subplot(4,2,3)
sns.distplot(df_trans['Annual_Rainfall'],bins = 10,color = 'fuchsia')
plt.subplot(4,2,4)
sns.distplot(df_trans['Fertilizer'],bins = 10, color = 'indigo')

plt.show()

"""## Viewing the Q-Q Plot after the Transformation"""

plt.figure(figsize=(15,20))
plt.subplot(4,2,1)
stats.probplot(df_trans['Area'], dist = 'norm', plot = plt)
plt.subplot(4,2,2)
stats.probplot(df_trans['Production'], dist = 'norm', plot = plt)
plt.subplot(4,2,3)
stats.probplot(df_trans['Annual_Rainfall'], dist = 'norm', plot = plt)
plt.subplot(4,2,4)
stats.probplot(df_trans['Fertilizer'], dist = 'norm', plot = plt)

plt.show()

"""# Linear Regression with skewed data"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred_train = lr.predict(x_train)
print("Training Accuracy : ",r2_score(y_train,y_pred_train))

y_pred_test = lr.predict(x_test)
print("Test Accuracy : ",r2_score(y_test,y_pred_test))

from sklearn.metrics import mean_absolute_error

# Assuming y_true contains the actual target values and y_pred contains the predicted values
mae = mean_absolute_error(y_train.to_numpy(),y_pred_train)
print("Mean Absolute Error:", mae)
print("Test Accuracy : ", mean_absolute_error(y_test.to_numpy(),y_pred_test))

# to store accuracy value
train_accu = []
test_accu = []

"""##  Linear Regression with Transformation Approach"""

lr.fit(x_train_transform1, y_train)

y_pred_train_ = lr.predict(x_train_transform1)
y_pred_test_ = lr.predict(x_test_transform1)

print("Training Accuracy : ",r2_score(y_train, y_pred_train_))
print()
print("Test Accuracy : ",r2_score(y_test, y_pred_test_))

train_accu.append(r2_score(y_train,y_pred_train_))
test_accu.append(r2_score(y_test,y_pred_test_))

from sklearn.metrics import mean_absolute_error

# Assuming y_true contains the actual target values and y_pred contains the predicted values
mae = mean_absolute_error(y_train.to_numpy(),y_pred_train_)
print("Mean Absolute Error:", mae)
print("Test Accuracy : ", mean_absolute_error(y_test.to_numpy(),y_pred_test_))

"""## Test Accuracy has improved after 'Yeo-Johnson' Transformation

### Here it is showing no case of overfitting or underfitting

## Variance Inflation Factor
"""

x1 = df_trans.copy()

from statsmodels.stats.outliers_influence import variance_inflation_factor

variable = x1

vif = pd.DataFrame()

vif['Variance Inflation Factor'] = [variance_inflation_factor(variable, i)
                                    for i in range(variable.shape[1])]

vif['Features'] = x1.columns

vif

"""VIF of the independent columns should be less than 5 to remove multicollinearity"""

x2 = x1.copy()

x2.drop(['Area'], axis = 1, inplace=True)

from statsmodels.stats.outliers_influence import variance_inflation_factor

variable = x2

vif = pd.DataFrame()

vif['Variance Inflation Factor'] = [variance_inflation_factor(variable, i)
                                    for i in range(variable.shape[1])]

vif['Features'] = x2.columns

vif

x2.drop(['Production'], axis = 1, inplace=True)

from statsmodels.stats.outliers_influence import variance_inflation_factor

variable = x2

vif = pd.DataFrame()

vif['Variance Inflation Factor'] = [variance_inflation_factor(variable, i)
                                    for i in range(variable.shape[1])]

vif['Features'] = x2.columns

vif

x2.head()

x_test1 = pd.DataFrame(x_test_transform1, columns=x_test.columns)
x_test1.drop(['Area','Production'], axis = 1, inplace = True)

# After applying vif
lr.fit(x2, y_train)

y_pred_train_ = lr.predict(x2)
y_pred_test_ = lr.predict(x_test1)

print("Training Accuracy : ",r2_score(y_train, y_pred_train_))
print()
print("Test Accuracy : ",r2_score(y_test, y_pred_test_))

train_accu.append(r2_score(y_train,y_pred_train_))
test_accu.append(r2_score(y_test,y_pred_test_))

from sklearn.metrics import mean_absolute_error

# Assuming y_true contains the actual target values and y_pred contains the predicted values
mae = mean_absolute_error(y_train.to_numpy(),y_pred_train_)
print("Mean Absolute Error:", mae)
print("Test Accuracy : ", mean_absolute_error(y_test.to_numpy(),y_pred_test_))

"""# Random Forest Regressor"""

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor()

regr.fit(x_train_transform1, y_train)

y_pred_train_regr= regr.predict(x_train_transform1)
y_pred_test_regr = regr.predict(x_test_transform1)

print("Training Accuracy : ",r2_score(y_train, y_pred_train_regr))
print("Test Accuracy : ",r2_score(y_test, y_pred_test_regr))

train_accu.append(r2_score(y_train,y_pred_train_regr))
test_accu.append(r2_score(y_test,y_pred_test_regr))

from sklearn.metrics import mean_absolute_error

# Assuming y_true contains the actual target values and y_pred contains the predicted values
mae = mean_absolute_error(y_train.to_numpy(),y_pred_train_regr)
print("Mean Absolute Error:", mae)
print("Test Accuracy : ", mean_absolute_error(y_test.to_numpy(),y_pred_test_regr))

# After applying vif
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor()


regr.fit(x2, y_train)

y_pred_train_regr= regr.predict(x2)
y_pred_test_regr = regr.predict(x_test1)

print("Training Accuracy : ",r2_score(y_train, y_pred_train_regr))
print("Test Accuracy : ",r2_score(y_test, y_pred_test_regr))

train_accu.append(r2_score(y_train,y_pred_train_regr))
test_accu.append(r2_score(y_test,y_pred_test_regr))

from sklearn.metrics import mean_absolute_error

# Assuming y_true contains the actual target values and y_pred contains the predicted values
mae = mean_absolute_error(y_train.to_numpy(),y_pred_train_regr)
print("Mean Absolute Error:", mae)
print("Test Accuracy : ", mean_absolute_error(y_test.to_numpy(),y_pred_test_regr))

from sklearn.metrics import mean_absolute_error

# Assuming y_true contains the actual target values and y_pred contains the predicted values
mae = mean_absolute_error(y_train.to_numpy(),y_pred_train_regr)
print("Mean Absolute Error:", mae)
print("Test Accuracy : ", mean_absolute_error(y_test.to_numpy(),y_pred_test_regr))

"""# CatBoostRegressor"""

!pip install catboost

from catboost import CatBoostRegressor
cat = CatBoostRegressor(learning_rate=0.15)
cat.fit(x_train_transform1, y_train)

y_pred_train_cat = cat.predict(x_train_transform1)
y_pred_test_cat = cat.predict(x_test_transform1)

print("Training Accuracy : ",r2_score(y_train, y_pred_train_cat))
print()
print("Test Accuracy : ",r2_score(y_test, y_pred_test_cat))

train_accu.append(r2_score(y_train,y_pred_train_cat))
test_accu.append(r2_score(y_test,y_pred_test_cat))

from sklearn.metrics import mean_absolute_error

# Assuming y_true contains the actual target values and y_pred contains the predicted values
mae = mean_absolute_error(y_train.to_numpy(),y_pred_train_cat)
print("Mean Absolute Error:", mae)
print("Test Accuracy : ", mean_absolute_error(y_test.to_numpy(),y_pred_test_cat))

# After applying vif
from catboost import CatBoostRegressor
cat = CatBoostRegressor(learning_rate=0.15)
cat.fit(x2, y_train)

y_pred_train_cat = cat.predict(x2)
y_pred_test_cat = cat.predict(x_test1)

print("Training Accuracy : ",r2_score(y_train, y_pred_train_cat))
print()
print("Test Accuracy : ",r2_score(y_test, y_pred_test_cat))

train_accu.append(r2_score(y_train,y_pred_train_cat))
test_accu.append(r2_score(y_test,y_pred_test_cat))

from sklearn.metrics import mean_absolute_error

# Assuming y_true contains the actual target values and y_pred contains the predicted values
mae = mean_absolute_error(y_train.to_numpy(),y_pred_train_cat)
print("Mean Absolute Error:", mae)
print("Test Accuracy : ", mean_absolute_error(y_test.to_numpy(),y_pred_test_cat))

"""# Comparison of the models"""

algorithm = ['LinearRegression','LRvif','RandomForestRegressor','RFRvif','CatBoostRegressor','CBRvif']
accu_data = {'Training Accuracy':train_accu,'Test Accuracy':test_accu}
model = pd.DataFrame(accu_data, index = algorithm)
model

accu_data

import matplotlib.pyplot as plt

# Given data
algorithm = ['LinearRegression', 'LRvif', 'RandomForestRegressor', 'RFRvif', 'SVRvif', 'CatBoostRegressor', 'CBRvif']
train_accu = [0.85, 0.87, 0.92, 0.91, 0.88, 0.93, 0.90]  # Example training accuracies
test_accu = [0.80, 0.82, 0.88, 0.87, 0.85, 0.89, 0.86]    # Example test accuracies

# Create the DataFrame
model = pd.DataFrame({'Training Accuracy': train_accu, 'Test Accuracy': test_accu}, index=algorithm)

# Plot the data
model.plot(kind='bar', figsize=(10, 6))
plt.title('Training and Test Accuracies for Different Models')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(loc='lower right')
plt.show()

# Box plot
model.plot(kind='box', figsize=(10, 6))
plt.title('Distribution of Accuracy Scores for Different Models')
plt.ylabel('Accuracy')
plt.show()

import matplotlib.pyplot as plt

# Given data
algorithm = ['LinearRegression', 'LRvif', 'RandomForestRegressor', 'RFRvif', 'CatBoostRegressor']
train_accu = [0.8567977421355504, 0.8513570270699844, 0.9966089814429742, 0.9952618563537046, 0.9996090425589377]
test_accu = [0.8201354232680313, 0.8106978418521673, 0.9825583500880688, 0.9781054037959107, 0.9679825446882119]

# Plot the data using line graphs
plt.figure(figsize=(10, 6))

# Plot training accuracy
plt.plot(algorithm, train_accu, marker='o', label='Training Accuracy')

# Plot test accuracy
plt.plot(algorithm, test_accu, marker='o', label='Test Accuracy')

plt.title('Training and Test Accuracies for Different Models')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Given data
algorithm = ['LinearRegression', 'LRvif', 'RandomForestRegressor', 'RFRvif', 'SVRvif', 'CatBoostRegressor']
train_accu = [0.8567977421355504, 0.8513570270699844, 0.9966089814429742, 0.9952618563537046, 0.9998265587624632, 0.9996090425589377]
test_accu = [0.8201354232680313, 0.8106978418521673, 0.9825583500880688, 0.9781054037959107, 0.9692481210303111, 0.9679825446882119]

# Create a DataFrame
data = pd.DataFrame({'Algorithm': algorithm * 2, 'Metric': ['Train Accuracy'] * 6 + ['Test Accuracy'] * 6,
                     'Accuracy': train_accu + test_accu})

# Plot the heatmap
plt.figure(figsize=(10, 6))
heatmap_data = data.pivot_table(index='Algorithm', columns='Metric', values='Accuracy', aggfunc='first')
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f")
plt.title('Training and Test Accuracies for Different Models')
plt.xlabel('Metric')
plt.ylabel('Algorithm')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Given data
algorithm = ['LinearRegression', 'LRvif', 'RandomForestRegressor', 'RFRvif', 'CatBoostRegressor']
train_accu = [0.8567977421355504, 0.8513570270699844, 0.9966089814429742, 0.9952618563537046, 0.9996090425589377]
test_accu = [0.8201354232680313, 0.8106978418521673, 0.9825583500880688, 0.9781054037959107, 0.9679825446882119]

# Create a DataFrame
data = pd.DataFrame({'Algorithm': algorithm, 'Train Accuracy': train_accu, 'Test Accuracy': test_accu})

# Set custom color palette
colors = sns.color_palette("pastel")

# Plot the table
plt.figure(figsize=(10, 5))
sns.set(style="whitegrid")
table = sns.heatmap(data.pivot_table(index='Algorithm', values=['Train Accuracy', 'Test Accuracy'], aggfunc='mean'),
                    annot=True, cmap="Blues", fmt=".3f", cbar=False)
plt.title('Training and Test Accuracies for Different Algorithms')
plt.xticks(rotation=45)
plt.show()

# Plot the bar diagram
plt.figure(figsize=(10, 6))
sns.barplot(x='Algorithm', y='Train Accuracy', data=data, palette=colors, label='Train Accuracy')
sns.barplot(x='Algorithm', y='Test Accuracy', data=data, palette=colors, label='Test Accuracy')
plt.title('Training and Test Accuracies for Different Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Given data
algorithm = ['LinearRegression', 'LRvif', 'RandomForestRegressor', 'RFRvif', 'SVRvif', 'CatBoostRegressor']
train_accu = [0.8567977421355504, 0.8513570270699844, 0.9966089814429742, 0.9952618563537046, 0.9998265587624632, 0.9996090425589377]
test_accu = [0.8201354232680313, 0.8106978418521673, 0.9825583500880688, 0.9781054037959107, 0.9692481210303111, 0.9679825446882119]

# Function to update the plot for each frame
def update(frame):
    ax.clear()
    ax.bar(algorithm[:frame+1], train_accu[:frame+1], label='Training Accuracy', color='b', alpha=0.7)
    ax.bar(algorithm[:frame+1], test_accu[:frame+1], label='Test Accuracy', color='r', alpha=0.7)
    ax.set_title('Accuracy Comparison')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Algorithm')
    ax.legend()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Animate the plot
ani = FuncAnimation(fig, update, frames=len(algorithm), interval=1000, repeat=False)

# Save the animation as a GIF
ani.save('accuracy_evolution.gif', writer='pillow', fps=1)

plt.show()

import matplotlib.pyplot as plt

# Data
models = ['Linear Regression', 'LRvif', 'RandomForestRegressor', 'RFRvif', 'CatBoostRegressor', 'CBRvif']
accuracy_test = [0.856798, 0.851357, 0.996493, 0.994876, 0.999827, 0.999609]
r2_test = [0.820135, 0.810698, 0.980769, 0.978354, 0.969248, 0.967983]
mse_train = [112404.84, 109394.72, 15086433.42, 1503124.20, 1527384.71, 1526666.23]
mse_test = [158486.85, 144114.66, 1567319.30, 1512510.82, 1544119.08, 1514511.91]
mae_train = [55.54, 76.61, 2.85, 3.83, 1.47, 2.13]
mae_test = [62.25, 82.64, 9.00, 9.51, 11.08, 12.53]

# Plot
plt.figure(figsize=(5, 6))

plt.plot(models, accuracy_test, marker='o', label='Accuracy (Test)', color='blue')
plt.title('Accuracy (Test)')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 6))
plt.plot(models, r2_test, marker='o', label='R2-value (Test)', color='green')
plt.title('R2-value (Test)')
plt.xlabel('Models')
plt.ylabel('R2-value')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 6))
plt.plot(models, mse_train, marker='o', label='MSE (Train)', color='red')
plt.title('MSE (Train)')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 6))
plt.plot(models, mse_test, marker='o', label='MSE (Test)', color='orange')
plt.title('MSE (Test)')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(5, 6))
plt.plot(models, mae_train, marker='o', label='MAE (Train)', color='purple')
plt.title('MAE (Train)')
plt.xlabel('Models')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(5, 6))
plt.plot(models, mae_test, marker='o', label='MAE (Test)', color='brown')
plt.title('MAE (Test)')
plt.xlabel('Models')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Data
models = ['Linear Regression', 'LRvif', 'RandomForestRegressor', 'RFRvif', 'CatBoostRegressor', 'CBRvif']
r2_train = [0.856798, 0.851357, 0.996493, 0.994876, 0.999827, 0.999609]
r2_test = [0.820135, 0.810698, 0.980769, 0.978354, 0.969248, 0.967983]
mse_train = [112404.84, 109394.72, 15086433.42, 1503124.20, 1527384.71, 1526666.23]
mse_test = [158486.85, 144114.66, 1567319.30, 1512510.82, 1544119.08, 1514511.91]
mae_train = [55.54, 76.61, 2.85, 3.83, 1.47, 2.13]
mae_test = [62.25, 82.64, 9.00, 9.51, 11.08, 12.53]

# Assign numeric indices to models
indices = range(len(models))

# Plot
fig, axs = plt.subplots(1, 3, figsize=(15, 7))

axs[0].plot(indices, r2_train, marker='o', color='skyblue',label="r2_train")
axs[0].plot(indices, r2_test, marker='o', color='salmon',label="r2_test")
axs[0].set_title('R-squared')
axs[0].legend()
axs[0].set_xticks(indices)
axs[0].set_xticklabels(models, rotation=45)

axs[1].plot(indices, mse_train, marker='o', color='lightgreen', label='MSE (Train)')
axs[1].plot(indices, mse_test, marker='o', color='lightcoral', label='MSE (Test)')
axs[1].set_title('Mean Squared Error (MSE)')
axs[1].legend()
axs[1].set_xticks(indices)
axs[1].set_xticklabels(models, rotation=45)

axs[2].plot(indices, mae_train, marker='o', color='lightblue', label='MAE (Train)')
axs[2].plot(indices, mae_test, marker='o', color='orange', label='MAE (Test)')
axs[2].set_title('Mean Absolute Error (MAE)')
axs[2].legend()
axs[2].set_xticks(indices)
axs[2].set_xticklabels(models, rotation=45)

plt.tight_layout()
plt.show()
