import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Feature engineering and preprocessing
def preprocess_data(data):
    # Fill missing numerical values with median
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col] = data[col].fillna(data[col].median())
    
    # Fill missing categorical values with mode
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # Encode categorical features
    label_encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    return data

# Preprocess the train and test data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Features and target variable
X = train_data.drop(['Id', 'SalePrice'], axis=1)
y = train_data['SalePrice']
X_test = test_data.drop(['Id'], axis=1)

# Split train data for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training using Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_valid_pred = model.predict(X_valid)

# Metrics
mse = mean_squared_error(y_valid, y_valid_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_valid, y_valid_pred)
r2 = r2_score(y_valid, y_valid_pred)
accuracy = 100 - (np.mean(np.abs((y_valid - y_valid_pred) / y_valid)) * 100)  # Accuracy as percentage

print("Validation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-Squared: {r2:.2f}")
print(f"Accuracy: {accuracy:.2f}%")

# Predict on test data
test_predictions = model.predict(X_test)

# Prepare submission file
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")