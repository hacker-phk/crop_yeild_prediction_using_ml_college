import matplotlib.pyplot as plt

# Data
models = ['Linear Regression', 'LRvif', 'RandomForestRegressor', 'RFRvif', 'CatBoostRegressor', 'CBRvif']
train_r2 = [0.856798, 0.851357, 0.996493, 0.994876, 0.999827, 0.999609]
test_r2 = [0.820135, 0.810698, 0.980769, 0.978354, 0.969248, 0.967983]
train_mse = [112404.84, 109394.72, 15086433.42, 1503124.20, 1527384.71, 1526666.23]
test_mse = [158486.85, 144114.66, 1567319.30, 1512510.82, 1544119.08, 1514511.91]
train_mae = [55.54, 76.61, 2.85, 3.83, 1.47, 2.13]
test_mae = [62.25, 82.64, 9.00, 9.51, 11.08, 12.53]

# Plot R2
plt.figure(figsize=(10, 6))
plt.bar(models, train_r2, alpha=0.5, label='Train R2')
plt.bar(models, test_r2, alpha=0.5, label='Test R2')
plt.title('R2 Values')
plt.xlabel('Models')
plt.ylabel('R2')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot MSE
plt.figure(figsize=(10, 6))
plt.bar(models, train_mse, alpha=0.5, label='Train MSE')
plt.bar(models, test_mse, alpha=0.5, label='Test MSE')
plt.title('Mean Squared Error (MSE)')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot MAE
plt.figure(figsize=(10, 6))
plt.bar(models, train_mae, alpha=0.5, label='Train MAE')
plt.bar(models, test_mae, alpha=0.5, label='Test MAE')
plt.title('Mean Absolute Error (MAE)')
plt.xlabel('Models')
plt.ylabel('MAE')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
