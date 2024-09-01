import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Load the dataset
file_path = r"C:\Users\Hemanth Kumar P\Desktop\final_no_more_than_75_percent_zeros.csv"
df = pd.read_csv(file_path)

# Feature Engineering

# Temporal Features
df['Month'] = pd.to_datetime(df['Year'], format='%Y').dt.month
df['Quarter'] = pd.to_datetime(df['Year'], format='%Y').dt.quarter

# Interaction Terms
df['Area_Fertilizer'] = df['Area'] * df['Fertilizer']
df['Rainfall_Temperature'] = df['Annual_Rainfall'] * df['Temperature']

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Temperature']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Temperature']))

# Concatenate the polynomial features to the original DataFrame
df = pd.concat([df, poly_df], axis=1)

# Calculate mean temperature for each unique 'Area'
mean_temp_area = df.groupby('Area')['Temperature'].mean().to_dict()
df['Temperature_Mean_Area'] = df['Area'].map(mean_temp_area)

# Calculate median rainfall for each unique 'Area'
median_rainfall_area = df.groupby('Area')['Annual_Rainfall'].median().to_dict()
df['Rainfall_Median_Area'] = df['Area'].map(median_rainfall_area)

# Target Encoding
df['State_Yield_Mean'] = df.groupby('State')['Yield'].transform('mean')

# Assuming 'Yield' is the target variable
X = df.drop(columns=['Yield'])  # Drop the target variable
y = df['Yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Lasso and Ridge models
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha parameter
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha parameter

# Train the models
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# Make predictions
lasso_predictions = lasso_model.predict(X_test)
ridge_predictions = ridge_model.predict(X_test)

# Combine predictions using simple averaging
ensemble_predictions = (lasso_predictions + ridge_predictions) / 2

# Evaluate the performance
ensemble_rmse = np.sqrt(np.mean((ensemble_predictions - y_test)**2))
print(f"Ensemble RMSE: {ensemble_rmse}")

# You can also compare the performance of individual models for reference
lasso_rmse = np.sqrt(np.mean((lasso_predictions - y_test)**2))
ridge_rmse = np.sqrt(np.mean((ridge_predictions - y_test)**2))

print(f"Lasso RMSE: {lasso_rmse}")
print(f"Ridge RMSE: {ridge_rmse}")
