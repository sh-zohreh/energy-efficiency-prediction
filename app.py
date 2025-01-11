import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap
import streamlit as st

def load_data(file_path):
    """
    Loads the dataset from the specified file path.
    
    Args:
        file_path (str): Path to the dataset file.
        
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(X):
    """
    Preprocesses the features by filling missing values and scaling the data.
    
    Args:
        X (pd.DataFrame): Input features.
        
    Returns:
        np.ndarray: Scaled feature data.
        StandardScaler: Fitted scaler for transforming new data.
    """
    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_and_evaluate_model(X, y, target_name):
    """
    Trains an XGBoost regression model and evaluates its performance.
    
    Args:
        X (np.ndarray): Input features.
        y (pd.Series): Target variable.
        target_name (str): Name of the target variable for logging.
        
    Returns:
        xgb.XGBRegressor: Trained model.
        float: Mean Squared Error (MSE) of the model.
        float: Mean Absolute Error (MAE) of the model.
        float: R-squared (R2) score of the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    return model, mse, mae, r2

def main():
    """
    Main function to run the Streamlit application for predicting energy efficiency.
    """
    st.title("Energy Efficiency Prediction")
    st.write("This application predicts Heating Load and Cooling Load based on building characteristics.")
    
    file_path = "data/data.csv"  # Update to match the data folder
    df = load_data(file_path)
    st.write("### Dataset Overview")
    st.dataframe(df.head())
    st.write(f"**Number of samples:** {df.shape[0]}")
    st.write(f"**Number of features:** {df.shape[1]}")

    X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    y_heating = df['Y1']
    y_cooling = df['Y2']
    X_scaled, scaler = preprocess_data(X)

    st.write("Training the model for Heating Load...")
    heating_model, heating_mse, heating_mae, heating_r2 = train_and_evaluate_model(X_scaled, y_heating, "Heating Load")
    st.write(f"**Heating Load Model Performance:** MSE={heating_mse:.2f}, MAE={heating_mae:.2f}, R2={heating_r2:.2f}")

    st.write("Training the model for Cooling Load...")
    cooling_model, cooling_mse, cooling_mae, cooling_r2 = train_and_evaluate_model(X_scaled, y_cooling, "Cooling Load")
    st.write(f"**Cooling Load Model Performance:** MSE={cooling_mse:.2f}, MAE={cooling_mae:.2f}, R2={cooling_r2:.2f}")

    st.write("### Make Predictions")
    st.write("Enter building characteristics below:")
    X1 = st.number_input("Relative Compactness (X1) [0.5 - 1.0]", min_value=0.5, max_value=1.0, value=0.8)
    X2 = st.number_input("Surface Area (X2) [400 - 900 m²]", min_value=400.0, max_value=900.0, value=500.0)
    X3 = st.number_input("Wall Area (X3) [200 - 500 m²]", min_value=200.0, max_value=500.0, value=300.0)
    X4 = st.number_input("Roof Area (X4) [100 - 300 m²]", min_value=100.0, max_value=300.0, value=150.0)
    X5 = st.number_input("Overall Height (X5) [3 - 10 m]", min_value=3.0, max_value=10.0, value=7.0)
    X6 = st.selectbox("Orientation (X6)", options=[2, 3, 4, 5], index=2, help="2=North, 3=East, 4=South, 5=West")
    X7 = st.number_input("Glazing Area (X7) [0.0 - 0.5]", min_value=0.0, max_value=0.5, value=0.1)
    X8 = st.selectbox("Glazing Area Distribution (X8)", options=[0, 1, 2, 3, 4, 5], index=1)
    
    input_data = np.array([[X1, X2, X3, X4, X5, X6, X7, X8]])
    input_data_scaled = scaler.transform(input_data)
    heating_prediction = heating_model.predict(input_data_scaled)
    cooling_prediction = cooling_model.predict(input_data_scaled)
    
    st.write(f"**Predicted Heating Load:** {heating_prediction[0]:.2f}")
    st.write(f"**Predicted Cooling Load:** {cooling_prediction[0]:.2f}")
    
    # SHAP explanations
    st.write("### Model Explanation using SHAP")
    explainer = shap.TreeExplainer(heating_model)
    shap_values = explainer.shap_values(input_data_scaled)
    st.write("#### SHAP Summary Plot for Heating Load")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_data, feature_names=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'], plot_type="bar", show=False)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
