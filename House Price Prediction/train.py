import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_data(filepath):
    """Loads the dataset from the CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please download it and place it in the project folder.")
    
    # housing data often has 'NA' strings for missing values
    df = pd.read_csv(filepath, na_values=['NA', 'na', 'NULL'])
    return df

def preprocess_data(df):
    """
    Performs data cleaning and preprocessing.
    """
    print("Initial Data Shape:", df.shape)
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing Values:\n", missing_counts[missing_counts > 0])
    else:
        print("No missing values found.")
    
    # Drop rows with missing values
    original_len = len(df)
    df = df.dropna()
    dropped_len = original_len - len(df)
    print(f"Dropped {dropped_len} rows with missing values. New shape: {df.shape}")
    
    if 'MEDV' in df.columns:
        target_col = 'MEDV'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        # Assume the last column is the target
        target_col = df.columns[-1]
        print(f"Assuming last column '{target_col}' is the target variable.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def train_models(X_train, X_test, y_train, y_test):
    """Trains Linear Regression and Random Forest models."""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train) 
    y_pred_rf = rf_model.predict(X_test)
    
    return (y_pred_lr, y_pred_rf)

def evaluate_model(name, y_test, y_pred):
    """Calculates and prints evaluation metrics."""
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print to console
    print(f"\n--- {name} Performance ---")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Append to results file
    with open("results.txt", "a") as f:
        f.write(f"\n--- {name} Performance ---\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
    
    return r2

def plot_results(y_test, y_pred_lr, y_pred_rf):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_lr, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Prices (MEDV)")
    plt.ylabel("Predicted Prices")
    plt.title("Linear Regression: Actual vs Predicted")
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Prices (MEDV)")
    plt.ylabel("Predicted Prices")
    plt.title("Random Forest: Actual vs Predicted")
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png')
    print("\nSaved prediction plot to 'prediction_comparison.png'")

def main():
    # Prioritize HousingData.csv
    possible_names = ['HousingData.csv', 'Boston.csv', 'housing.csv', 'boston_housing.csv', 'data.csv']
    dataset_path = None
    
    for name in possible_names:
        if os.path.exists(name):
            dataset_path = name
            break
            
    if dataset_path is None:
        print("Error: Could not find dataset file.")
        print(f"Please place one of {possible_names} in the current directory.")
        return

    print(f"Loading data from {dataset_path}...")
    df = load_data(dataset_path)
    
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred_lr, y_pred_rf = train_models(X_train, X_test, y_train, y_test)
    
    evaluate_model("Linear Regression", y_test, y_pred_lr)
    evaluate_model("Random Forest", y_test, y_pred_rf)
    
    plot_results(y_test, y_pred_lr, y_pred_rf)

if __name__ == "__main__":
    main()
