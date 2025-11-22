"""
Utility functions for data preprocessing and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    def __init__(self, scaling_method='minmax', sequence_length=60):
        """
        Initialize data preprocessor
        
        Args:
            scaling_method: 'minmax' or 'standard'
            sequence_length: Number of time steps for sequence models (LSTM, GRU)
        """
        self.scaling_method = scaling_method
        self.sequence_length = sequence_length
        self.scaler_X = None
        self.scaler_y = None
        self.feature_columns = None
        
    def create_sequences(self, data, target):
        """Create sequences for LSTM/GRU models"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def prepare_data_for_lstm(self, df, target_col='Close', test_size=0.2, val_size=0.1):
        """
        Prepare data for LSTM/GRU models with sequences
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, dates_test
        """
        # Select features (exclude target and date-related columns)
        exclude_cols = [target_col, 'Adj Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        
        # Prepare features and target
        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)
        dates = df.index
        
        # Initialize scalers
        if self.scaling_method == 'minmax':
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        dates_seq = dates[self.sequence_length:]
        
        # Split data
        train_size = int(len(X_seq) * (1 - test_size - val_size))
        val_size_abs = int(len(X_seq) * val_size)
        
        X_train = X_seq[:train_size]
        y_train = y_seq[:train_size]
        
        X_val = X_seq[train_size:train_size + val_size_abs]
        y_val = y_seq[train_size:train_size + val_size_abs]
        
        X_test = X_seq[train_size + val_size_abs:]
        y_test = y_seq[train_size + val_size_abs:]
        dates_test = dates_seq[train_size + val_size_abs:]
        
        print(f"\nData prepared for sequence models:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Features: {len(feature_cols)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, dates_test
    
    def prepare_data_for_ml(self, df, target_col='Close', test_size=0.2, val_size=0.1):
        """
        Prepare data for traditional ML models (XGBoost, Prophet)
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, dates_test
        """
        # Select features
        exclude_cols = [target_col, 'Adj Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values
        dates = df.index
        
        # Initialize scalers
        if self.scaling_method == 'minmax':
            self.scaler_X = MinMaxScaler()
        else:
            self.scaler_X = StandardScaler()
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Split data
        train_size = int(len(X_scaled) * (1 - test_size - val_size))
        val_size_abs = int(len(X_scaled) * val_size)
        
        X_train = X_scaled[:train_size]
        y_train = y[:train_size]
        
        X_val = X_scaled[train_size:train_size + val_size_abs]
        y_val = y[train_size:train_size + val_size_abs]
        
        X_test = X_scaled[train_size + val_size_abs:]
        y_test = y[train_size + val_size_abs:]
        dates_test = dates[train_size + val_size_abs:]
        
        print(f"\nData prepared for ML models:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Features: {len(feature_cols)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, dates_test
    
    def prepare_data_for_prophet(self, df, target_col='Close'):
        """
        Prepare data specifically for Facebook Prophet
        
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df[target_col].values
        })
        
        # Add additional regressors
        regressor_cols = ['Volume', 'SPX_Close', 'RSI']
        for col in regressor_cols:
            if col in df.columns:
                prophet_df[col] = df[col].values
        
        return prophet_df
    
    def save_scalers(self, save_dir='models/scalers/'):
        """Save fitted scalers"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, f"{save_dir}scaler_X.pkl")
        if self.scaler_y is not None:
            joblib.dump(self.scaler_y, f"{save_dir}scaler_y.pkl")
        
        # Save feature columns
        joblib.dump(self.feature_columns, f"{save_dir}feature_columns.pkl")
        
        print(f"✓ Scalers saved to {save_dir}")
    
    def load_scalers(self, save_dir='models/scalers/'):
        """Load fitted scalers"""
        self.scaler_X = joblib.load(f"{save_dir}scaler_X.pkl")
        try:
            self.scaler_y = joblib.load(f"{save_dir}scaler_y.pkl")
        except:
            self.scaler_y = None
        
        self.feature_columns = joblib.load(f"{save_dir}feature_columns.pkl")
        
        print(f"✓ Scalers loaded from {save_dir}")


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional Accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }


def print_metrics(metrics, model_name="Model"):
    """Print evaluation metrics in a formatted way"""
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")
    print(f"  RMSE (Root Mean Squared Error): ${metrics['RMSE']:.2f}")
    print(f"  MAE (Mean Absolute Error):      ${metrics['MAE']:.2f}")
    print(f"  MAPE (Mean Abs % Error):        {metrics['MAPE']:.2f}%")
    print(f"  R² Score:                       {metrics['R2']:.4f}")
    print(f"  Directional Accuracy:           {metrics['Directional_Accuracy']:.2f}%")
    print(f"{'='*60}\n")
