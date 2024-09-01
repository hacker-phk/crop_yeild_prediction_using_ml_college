import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the dataset
file_path = r"C:\Users\Hemanth Kumar P\Desktop\final_no_more_than_75_percent_zeros.csv"
df = pd.read_csv(file_path)

# Keep only specified columns
columns_to_keep = ['Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield', 'Temperature', 'Crop']
df = df[columns_to_keep]

# Encode the 'Crop' column using LabelEncoder
label_encoder = LabelEncoder()
df['Crop'] = label_encoder.fit_transform(df['Crop'])

# Assuming 'Yield' is the target variable
X = df[['Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Temperature', 'Crop']]
y = df['Yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest and Gradient Boosting models
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)  
gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42)  

# Train the models
random_forest_model.fit(X_train, y_train)
gradient_boosting_model.fit(X_train, y_train)

# Make predictions
random_forest_predictions = random_forest_model.predict(X_test)
gradient_boosting_predictions = gradient_boosting_model.predict(X_test)

# Combine predictions using simple averaging
ensemble_predictions_rf_gb = (random_forest_predictions + gradient_boosting_predictions) / 2

# Evaluate the performance
ensemble_rmse_rf_gb = np.sqrt(np.mean((ensemble_predictions_rf_gb - y_test)**2))
print(f"Ensemble (Random Forest + Gradient Boosting) RMSE: {ensemble_rmse_rf_gb}")

# Define a threshold for accuracy
threshold = 0.1

# Calculate accuracy for Random Forest and Gradient Boosting
random_forest_accuracy = np.mean(np.abs(random_forest_predictions - y_test) <= threshold)
gradient_boosting_accuracy = np.mean(np.abs(gradient_boosting_predictions - y_test) <= threshold)
ensemble_accuracy = np.mean(np.abs(ensemble_predictions_rf_gb - y_test) <= threshold)

# Print accuracy percentages
print(f"Random Forest Accuracy: {random_forest_accuracy * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {gradient_boosting_accuracy * 100:.2f}%")
print(f"Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")
