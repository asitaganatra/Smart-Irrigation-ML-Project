import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

# Define a unified set of features for comparison across datasets
FEATURE_MAPPING = {
    # Real Data Feature Name: Unified Name for processing
    'crop_name': 'crop_type',
    'soil_type': 'soil_type',
    'temperature_C': 'temperature',
    'humidity_%': 'humidity',
    'rainfall_mm': 'rainfall',
    'wind_speed_m_s': 'wind_speed',
    'soil_moisture_%': 'soil_moisture'
}
TARGET_COL = 'water_req_target'
REAL_DATA_FILE = 'Smart_irrigation_dataset.csv'

class SmartIrrigationModel:
    """
    Handles model initialization, data loading, preprocessing, and comparative training.
    Supports Random Forest Regressor and Decision Tree Regressor.
    """
    def __init__(self, model_type='RandomForest'):
        if model_type == 'DecisionTree':
            self.model = DecisionTreeRegressor(random_state=42)
        else:
            self.model = RandomForestRegressor(random_state=42)
            
        self.model_type = model_type
        self.expected_features = None

    def generate_synthetic_data(self, n_samples=3000):
        """Generates synthetic data (Dataset 1) with unified feature names."""
        crops = ['Wheat', 'Rice', 'Corn', 'Tomato', 'Potato', 'Cotton']
        soil_types = ['Clay', 'Sandy', 'Loamy', 'Silty']
        
        data = {
            'crop_type': np.random.choice(crops, n_samples),
            'soil_type': np.random.choice(soil_types, n_samples),
            'temperature': np.random.uniform(10, 40, n_samples),
            'humidity': np.random.uniform(40, 90, n_samples),
            'rainfall': np.random.uniform(0, 30, n_samples),
            'wind_speed': np.random.uniform(1, 10, n_samples),
            'soil_moisture': np.random.uniform(30, 70, n_samples),
        }
        df = pd.DataFrame(data)
        
        # Target variable (Synthetic Water Requirement: L/m²)
        df[TARGET_COL] = (
            50 + (df['temperature'] * 0.8) + (df['humidity'] * -0.3) + 
            (df['rainfall'] * -1.5) + (df['soil_moisture'] * -0.6) +
            np.random.normal(0, 8, n_samples)
        )
        df[TARGET_COL] = df[TARGET_COL].clip(lower=0)
        return df

    def load_and_clean_real_data(self, filename=REAL_DATA_FILE):
        """Loads and cleans the real-world data (Dataset 2) for unified training."""
        df = pd.read_csv(filename)
        df.rename(columns=FEATURE_MAPPING, inplace=True)
        # Use the real irrigation amount as the target
        df[TARGET_COL] = df['irrigation_amount_m3']
        
        # Select only the unified features and the target column
        features_and_target = list(FEATURE_MAPPING.values()) + [TARGET_COL]
        return df[features_and_target].copy()

    def preprocess_data(self, df):
        """One-hot encodes categorical features and splits data into X and y."""
        X = df.drop(TARGET_COL, axis=1)
        y = df[TARGET_COL]
        categorical_cols = ['crop_type', 'soil_type']
        # Use drop_first=True to avoid multicollinearity
        X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        return X_processed, y, list(X_processed.columns)

    def train_and_evaluate_model(self, model_instance, df):
        """Trains a given model instance and calculates metrics."""
        X, y, features = self.preprocess_data(df)
        model_instance.expected_features = features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_instance.model.fit(X_train, y_train)
        y_pred = model_instance.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return model_instance.model_type, mae, rmse, r2

    def comparative_training(self):
        """Performs training and comparison for the assignment requirement."""
        df_synthetic = self.generate_synthetic_data()
        df_real = self.load_and_clean_real_data()

        datasets = {
            'Synthetic Data': df_synthetic,
            f'Real-World Data ({REAL_DATA_FILE})': df_real
        }
        results = []
        
        for ds_name, df in datasets.items():
            # Create a clean file name for saving the models
            ds_name_safe = ds_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").lower()
            
            # --- Model 1: Random Forest ---
            rf_model = SmartIrrigationModel(model_type='RandomForest')
            rf_name, rf_mae, rf_rmse, rf_r2 = self.train_and_evaluate_model(rf_model, df)
            rf_model.save_model(f'rf_model_{ds_name_safe}.pkl')
            results.append({'Dataset': ds_name, 'Model': rf_name, 'MAE': rf_mae, 'RMSE': rf_rmse, 'R2': rf_r2})
            
            # --- Model 2: Decision Tree ---
            dt_model = SmartIrrigationModel(model_type='DecisionTree')
            dt_name, dt_mae, dt_rmse, dt_r2 = self.train_and_evaluate_model(dt_model, df)
            dt_model.save_model(f'dt_model_{ds_name_safe}.pkl')
            results.append({'Dataset': ds_name, 'Model': dt_name, 'MAE': dt_mae, 'RMSE': dt_rmse, 'R2': dt_r2})
            
        return pd.DataFrame(results)

    # --- Saving/Loading (Robust for joblib) ---
    def save_model(self, filename):
        """Saves the trained model and feature list."""
        # Saves the core model object and the list of features
        joblib.dump((self.model, self.expected_features), filename)
    
    def load_model(self, filename):
        """Loads the trained model and feature list."""
        self.model, self.expected_features = joblib.load(filename)
        if isinstance(self.model, DecisionTreeRegressor):
            self.model_type = 'DecisionTree'
        elif isinstance(self.model, RandomForestRegressor):
            self.model_type = 'RandomForest'

    def predict(self, input_data):
        """Preprocesses input data using expected features and makes a prediction."""
        # Convert app input to DataFrame and rename to unified features
        input_df = pd.DataFrame([input_data])
        
        categorical_cols = ['crop_type', 'soil_type']
        # One-hot encode the input, matching training configuration (drop_first=True)
        input_processed = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # Ensure all expected columns are present (critical for deployment)
        for col in self.expected_features:
            if col not in input_processed.columns:
                input_processed[col] = 0
        
        # Reorder columns to match training data
        input_processed = input_processed[self.expected_features]
        
        # Prediction output is in m³
        return self.model.predict(input_processed)[0]

    def get_recommendations(self, input_data, prediction):
        """Generates human-readable recommendations and warnings."""
        # Prediction is in m³
        recommendations = [
            f"Irrigate with {prediction:.2f} m³ of water",
            f"The predicted volume is high, consider applying irrigation in multiple smaller doses.",
            "Always check the 24-hour weather forecast for unexpected rainfall."
        ]
        
        warnings = []
        
        # Checks use the original input_data names from the Streamlit app
        if input_data.get('temperature_C', 0) > 35:
            warnings.append("⚠️ Extreme heat detected. Consider shade nets and increase irrigation frequency.")
        if input_data.get('soil_moisture_%', 0) > 80:
            warnings.append("⚠️ High soil moisture. Reduce watering to avoid waterlogging.")
        if input_data.get('ph_level', 6.5) < 5.5 or input_data.get('ph_level', 6.5) > 8.0:
            warnings.append("⚠️ Soil pH is outside optimal range. Consider soil treatment.")
            
        return recommendations, warnings

if __name__ == "__main__":
    # This block generates all models and the comparative results file locally.
    print("Starting Comparative Training for Assignment Submission...")
    trainer = SmartIrrigationModel() 
    results_df = trainer.comparative_training() 
    results_df.to_csv('comparative_results.csv', index=False)
    print("\nTraining and Evaluation Complete. The following models and results were generated:")
    print(results_df)
    print(f"\nSaved models: rf_model_synthetic_data.pkl, dt_model_synthetic_data.pkl, rf_model_real-world_data_{REAL_DATA_FILE.replace('.', '').lower()}.pkl, dt_model_real-world_data_{REAL_DATA_FILE.replace('.', '').lower()}.pkl")
    print(f"**Action Required:** Ensure the best model ({'rf_model_real-world_data_smart_irrigation_datasetcsv.pkl'}) is available for the app to load.")
