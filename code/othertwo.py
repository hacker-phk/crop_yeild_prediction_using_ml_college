import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import r2_score


# Load the dataset
file_path = r"D:\interview final\college project\datasets\final_no_more_than_75_percent_zeros.csv"
df = pd.read_csv(file_path)

# Assuming 'Yield' is the target variable
X = df[['Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Temperature', 'season_encoded', 'rain_fertilizer_interaction', 'temp_mean', 'temp_std', 'temp_median', 'rainfall_mean', 'rainfall_std', 'rainfall_median']]

# Convert string columns to integers using LabelEncoder
label_encoder = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])

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
ensemble_rmse_rf_gb = np.sqrt(np.mean((ensemble_predictions_rf_gb - y_test) ** 2))
print(f"Ensemble (Random Forest + Gradient Boosting) RMSE: {ensemble_rmse_rf_gb}")

# Define a threshold for accuracy (you can adjust this based on your specific requirements)
threshold = 0.1  # For example, consider predictions within 0.1 of the true value as correct

# Define a function to calculate accuracy based on the threshold
def calculate_accuracy(predictions, true_values, threshold):
    correct_predictions = np.abs(predictions - true_values) <= threshold
    accuracy = np.mean(correct_predictions)
    return accuracy * 100

# Calculate accuracy for Random Forest and Gradient Boosting
random_forest_accuracy = calculate_accuracy(random_forest_predictions, y_test, threshold)
gradient_boosting_accuracy = calculate_accuracy(gradient_boosting_predictions, y_test, threshold)
ensemble_accuracy = calculate_accuracy(ensemble_predictions_rf_gb, y_test, threshold)

# Print accuracy percentages
print(f"Random Forest Accuracy: {random_forest_accuracy:.2f}%")
print(f"Gradient Boosting Accuracy: {gradient_boosting_accuracy:.2f}%")
print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}%")

# Calculate R-squared for Random Forest
random_forest_r2 = r2_score(y_test, random_forest_predictions)

# Calculate R-squared for Gradient Boosting
gradient_boosting_r2 = r2_score(y_test, gradient_boosting_predictions)

# Calculate R-squared for Ensemble
ensemble_r2 = r2_score(y_test, ensemble_predictions_rf_gb)

# Print R-squared values
print(f"Random Forest R-squared: {random_forest_r2:.4f}")
print(f"Gradient Boosting R-squared: {gradient_boosting_r2:.4f}")
print(f"Ensemble R-squared: {ensemble_r2:.4f}")
