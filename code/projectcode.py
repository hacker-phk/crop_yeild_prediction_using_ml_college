import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor

# Load the dataset from Excel
dataset_path = r"D:\interview final\college project\datasets\newdataset.xlsx"
df = pd.read_excel(dataset_path)

# Feature Engineering: Creating Interaction Features
df['rainfall_temperature_interaction'] = df['rainfall'] * df['temperature']
df['nutrient_ratio'] = df['N'] / (df['P'] + df['K'])

# Select features and target variable
features = df[['N', 'P', 'K', 'pH', 'rainfall', 'temperature', 'Area_in_hectares', 'rainfall_temperature_interaction', 'nutrient_ratio']]
target = df['Yield_ton_per_hec']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Model for feature selection
rf_feature_selection_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_feature_selection_model.fit(X_train_scaled, y_train)

# Use feature importance from Random Forest for feature selection
feature_importance = rf_feature_selection_model.feature_importances_
selected_features_indices = feature_importance.argsort()[-5:][::-1]  # Select top 5 features
selected_features = X_train.columns[selected_features_indices]

# Use only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Neural Network Model
def create_nn_model():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_selected.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Wrap the Keras model for scikit-learn compatibility
keras_regressor = KerasRegressor(build_fn=create_nn_model, epochs=50, batch_size=32, verbose=0)

# Gradient Boosting Model for prediction
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_selected, y_train)

# Create a stacked model with Random Forest, Neural Network, and Gradient Boosting
stacked_model = StackingRegressor(
    estimators=[
        ('rf', rf_feature_selection_model),
        ('gb.puml', keras_regressor),
        ('nn', gb_model)
    ],
    final_estimator=RidgeCV()
)

# Train the stacked model
stacked_model.fit(X_train_selected, y_train)

# Predictions using the stacked model
stacked_predictions = stacked_model.predict(X_test_selected)

# Evaluate the stacked model
stacked_mse = mean_squared_error(y_test, stacked_predictions)
stacked_r2 = r2_score(y_test, stacked_predictions)

print("Stacked Model (Random Forest for feature selection + Neural Network + Gradient Boosting):")
print(f'Mean Squared Error: {stacked_mse}')
print(f'R-squared: {stacked_r2}')
