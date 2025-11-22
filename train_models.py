"""
Main Training Script - Trains all 5 AI models for Tesla stock prediction
Self-contained version - no external module dependencies
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import joblib

# ============================================================
# INLINE: Data Preprocessing Functions
# ============================================================

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class DataPreprocessor:
    def __init__(self, scaling_method='minmax', sequence_length=60):
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
        """Prepare data for LSTM/GRU models with sequences"""
        exclude_cols = [target_col, 'Adj Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)
        dates = df.index
        
        if self.scaling_method == 'minmax':
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        dates_seq = dates[self.sequence_length:]
        
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
        """Prepare data for traditional ML models (XGBoost)"""
        exclude_cols = [target_col, 'Adj Close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values
        dates = df.index
        
        if self.scaling_method == 'minmax':
            self.scaler_X = MinMaxScaler()
        else:
            self.scaler_X = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        
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
    
    def save_scalers(self, save_dir='models/scalers/'):
        os.makedirs(save_dir, exist_ok=True)
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, f"{save_dir}scaler_X.pkl")
        if self.scaler_y is not None:
            joblib.dump(self.scaler_y, f"{save_dir}scaler_y.pkl")
        joblib.dump(self.feature_columns, f"{save_dir}feature_columns.pkl")
        print(f"✓ Scalers saved to {save_dir}")


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    
    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
        'MAPE': mape, 'Directional_Accuracy': directional_accuracy
    }


def print_metrics(metrics, model_name="Model"):
    print(f"\n{'='*60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*60}")
    print(f"  RMSE (Root Mean Squared Error): ${metrics['RMSE']:.2f}")
    print(f"  MAE (Mean Absolute Error):      ${metrics['MAE']:.2f}")
    print(f"  MAPE (Mean Abs % Error):        {metrics['MAPE']:.2f}%")
    print(f"  R² Score:                       {metrics['R2']:.4f}")
    print(f"  Directional Accuracy:           {metrics['Directional_Accuracy']:.2f}%")
    print(f"{'='*60}\n")


# ============================================================
# INLINE: LSTM Model
# ============================================================

def build_lstm_model(sequence_length, n_features, lstm_units=[128, 64, 32], 
                     dropout_rate=0.2, learning_rate=0.001):
    """Build LSTM model"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential(name='LSTM_Stock_Predictor')
    
    model.add(LSTM(units=lstm_units[0], return_sequences=True,
                   input_shape=(sequence_length, n_features), name='LSTM_1'))
    model.add(BatchNormalization(name='BN_1'))
    model.add(Dropout(dropout_rate, name='Dropout_1'))
    
    model.add(LSTM(units=lstm_units[1], return_sequences=True, name='LSTM_2'))
    model.add(BatchNormalization(name='BN_2'))
    model.add(Dropout(dropout_rate, name='Dropout_2'))
    
    model.add(LSTM(units=lstm_units[2], return_sequences=False, name='LSTM_3'))
    model.add(BatchNormalization(name='BN_3'))
    model.add(Dropout(dropout_rate, name='Dropout_3'))
    
    model.add(Dense(32, activation='relu', name='Dense_1'))
    model.add(Dropout(dropout_rate / 2, name='Dropout_4'))
    model.add(Dense(16, activation='relu', name='Dense_2'))
    model.add(Dense(1, name='Output'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    
    return model


# ============================================================
# INLINE: GRU Model
# ============================================================

def build_gru_model(sequence_length, n_features, gru_units=[128, 64, 32], 
                    dropout_rate=0.2, learning_rate=0.001):
    """Build GRU model"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential(name='GRU_Stock_Predictor')
    
    model.add(GRU(units=gru_units[0], return_sequences=True,
                  input_shape=(sequence_length, n_features), name='GRU_1'))
    model.add(BatchNormalization(name='BN_1'))
    model.add(Dropout(dropout_rate, name='Dropout_1'))
    
    model.add(GRU(units=gru_units[1], return_sequences=True, name='GRU_2'))
    model.add(BatchNormalization(name='BN_2'))
    model.add(Dropout(dropout_rate, name='Dropout_2'))
    
    model.add(GRU(units=gru_units[2], return_sequences=False, name='GRU_3'))
    model.add(BatchNormalization(name='BN_3'))
    model.add(Dropout(dropout_rate, name='Dropout_3'))
    
    model.add(Dense(32, activation='relu', name='Dense_1'))
    model.add(Dropout(dropout_rate / 2, name='Dropout_4'))
    model.add(Dense(16, activation='relu', name='Dense_2'))
    model.add(Dense(1, name='Output'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    
    return model


# ============================================================
# INLINE: Transformer Model
# ============================================================

def build_transformer_model(sequence_length, n_features, head_size=256, num_heads=4, 
                            ff_dim=4, num_transformer_blocks=2, mlp_units=[128], 
                            dropout=0.2, learning_rate=0.001):
    """Build Transformer model"""
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    
    inputs = keras.Input(shape=(sequence_length, n_features))
    x = inputs
    
    # Simple transformer-like architecture using Dense layers
    for _ in range(num_transformer_blocks):
        # Self-attention approximation with Dense layers
        attention = layers.Dense(head_size, activation='relu')(x)
        attention = layers.Dropout(dropout)(attention)
        x = layers.Add()([x, layers.Dense(n_features)(attention)])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed forward
        ff = layers.Dense(head_size * ff_dim, activation='relu')(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(n_features)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # MLP layers
    for dim in mlp_units:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs, name='Transformer_Stock_Predictor')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    
    return model


# ============================================================
# INLINE: XGBoost Model
# ============================================================

def build_xgboost_model(n_estimators=1000, learning_rate=0.01, max_depth=7,
                        subsample=0.8, colsample_bytree=0.8):
    """Build XGBoost model"""
    import xgboost as xgb
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    return model


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_all_models(data_path='data/tesla_spx500_complete.csv'):
    """Train all models and save results"""
    
    print("\n" + "="*80)
    print(" TESLA STOCK PREDICTION - TRAINING ALL MODELS")
    print("="*80 + "\n")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
        return None, None
    
    # Load data
    print("Loading data...")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Please run 'python collect_data.py' first!")
        return None, None
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
    print(f"✓ Features: {len(df.columns)}\n")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize results storage
    all_results = {}
    all_predictions = {}
    
    # Callbacks for training
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    # =============================================================
    # PREPARE DATA FOR SEQUENCE MODELS
    # =============================================================
    print("\n" + "="*80)
    print("PREPARING DATA FOR SEQUENCE MODELS (LSTM, GRU, Transformer)")
    print("="*80)
    
    preprocessor_seq = DataPreprocessor(scaling_method='minmax', sequence_length=60)
    X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq, dates_test = \
        preprocessor_seq.prepare_data_for_lstm(df, target_col='Close')
    
    preprocessor_seq.save_scalers('models/scalers_seq/')
    n_features = X_train_seq.shape[2]
    
    # =============================================================
    # MODEL 1: LSTM
    # =============================================================
    print("\n" + "="*80)
    print("TRAINING MODEL 1: LSTM (Long Short-Term Memory)")
    print("="*80)
    
    lstm_model = build_lstm_model(60, n_features)
    print("\nLSTM Model Summary:")
    lstm_model.summary()
    
    print("\nTraining LSTM model...")
    lstm_history = lstm_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50, batch_size=32, callbacks=callbacks, verbose=1
    )
    
    # Predictions
    y_pred_lstm_scaled = lstm_model.predict(X_test_seq, verbose=0)
    y_pred_lstm = preprocessor_seq.scaler_y.inverse_transform(y_pred_lstm_scaled)
    y_test_lstm = preprocessor_seq.scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
    
    lstm_metrics = calculate_metrics(y_test_lstm.flatten(), y_pred_lstm.flatten())
    print_metrics(lstm_metrics, "LSTM")
    
    lstm_model.save('models/lstm_model.keras')
    all_results['LSTM'] = lstm_metrics
    all_predictions['LSTM'] = {
        'y_true': y_test_lstm.flatten(),
        'y_pred': y_pred_lstm.flatten(),
        'dates': dates_test
    }
    print("✓ LSTM model saved to models/lstm_model.keras")
    
    # =============================================================
    # MODEL 2: GRU
    # =============================================================
    print("\n" + "="*80)
    print("TRAINING MODEL 2: GRU (Gated Recurrent Unit)")
    print("="*80)
    
    gru_model = build_gru_model(60, n_features)
    print("\nGRU Model Summary:")
    gru_model.summary()
    
    print("\nTraining GRU model...")
    gru_history = gru_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50, batch_size=32, callbacks=callbacks, verbose=1
    )
    
    y_pred_gru_scaled = gru_model.predict(X_test_seq, verbose=0)
    y_pred_gru = preprocessor_seq.scaler_y.inverse_transform(y_pred_gru_scaled)
    
    gru_metrics = calculate_metrics(y_test_lstm.flatten(), y_pred_gru.flatten())
    print_metrics(gru_metrics, "GRU")
    
    gru_model.save('models/gru_model.keras')
    all_results['GRU'] = gru_metrics
    all_predictions['GRU'] = {
        'y_true': y_test_lstm.flatten(),
        'y_pred': y_pred_gru.flatten(),
        'dates': dates_test
    }
    print("✓ GRU model saved to models/gru_model.keras")
    
    # =============================================================
    # MODEL 3: Transformer
    # =============================================================
    print("\n" + "="*80)
    print("TRAINING MODEL 3: Transformer")
    print("="*80)
    
    transformer_model = build_transformer_model(60, n_features)
    print("\nTransformer Model Summary:")
    transformer_model.summary()
    
    print("\nTraining Transformer model...")
    transformer_history = transformer_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50, batch_size=32, callbacks=callbacks, verbose=1
    )
    
    y_pred_transformer_scaled = transformer_model.predict(X_test_seq, verbose=0)
    y_pred_transformer = preprocessor_seq.scaler_y.inverse_transform(y_pred_transformer_scaled)
    
    transformer_metrics = calculate_metrics(y_test_lstm.flatten(), y_pred_transformer.flatten())
    print_metrics(transformer_metrics, "Transformer")
    
    transformer_model.save('models/transformer_model.keras')
    all_results['Transformer'] = transformer_metrics
    all_predictions['Transformer'] = {
        'y_true': y_test_lstm.flatten(),
        'y_pred': y_pred_transformer.flatten(),
        'dates': dates_test
    }
    print("✓ Transformer model saved to models/transformer_model.keras")
    
    # =============================================================
    # PREPARE DATA FOR ML MODELS
    # =============================================================
    print("\n" + "="*80)
    print("PREPARING DATA FOR ML MODELS (XGBoost)")
    print("="*80)
    
    preprocessor_ml = DataPreprocessor(scaling_method='standard', sequence_length=60)
    X_train_ml, X_val_ml, X_test_ml, y_train_ml, y_val_ml, y_test_ml, dates_test_ml = \
        preprocessor_ml.prepare_data_for_ml(df, target_col='Close')
    
    preprocessor_ml.save_scalers('models/scalers_ml/')
    
    # =============================================================
    # MODEL 4: XGBoost
    # =============================================================
    print("\n" + "="*80)
    print("TRAINING MODEL 4: XGBoost (Extreme Gradient Boosting)")
    print("="*80)
    
    xgb_model = build_xgboost_model()
    
    print("\nTraining XGBoost model...")
    xgb_model.fit(
        X_train_ml, y_train_ml,
        eval_set=[(X_train_ml, y_train_ml), (X_val_ml, y_val_ml)],
        verbose=100
    )
    
    y_pred_xgb = xgb_model.predict(X_test_ml)
    
    xgb_metrics = calculate_metrics(y_test_ml, y_pred_xgb)
    print_metrics(xgb_metrics, "XGBoost")
    
    # Feature importance
    importance = xgb_model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': preprocessor_ml.feature_columns,
        'importance': importance
    }).sort_values('importance', ascending=False).head(20)
    print("\nTop 20 Most Important Features:")
    print(feature_importance.to_string(index=False))
    
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    all_results['XGBoost'] = xgb_metrics
    all_predictions['XGBoost'] = {
        'y_true': y_test_ml,
        'y_pred': y_pred_xgb,
        'dates': dates_test_ml
    }
    print("\n✓ XGBoost model saved to models/xgboost_model.pkl")
    
    # =============================================================
    # MODEL 5: Simple Moving Average Baseline (instead of Prophet for simplicity)
    # =============================================================
    print("\n" + "="*80)
    print("TRAINING MODEL 5: Moving Average Baseline")
    print("="*80)
    
    # Simple baseline using moving average
    y_pred_ma = df['Close'].rolling(window=20).mean().iloc[-len(y_test_ml):].values
    
    ma_metrics = calculate_metrics(y_test_ml, y_pred_ma)
    print_metrics(ma_metrics, "Moving Average Baseline")
    
    all_results['MA_Baseline'] = ma_metrics
    all_predictions['MA_Baseline'] = {
        'y_true': y_test_ml,
        'y_pred': y_pred_ma,
        'dates': dates_test_ml
    }
    
    # =============================================================
    # COMPARE ALL MODELS
    # =============================================================
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.sort_values('RMSE')
    
    print(comparison_df.to_string())
    
    comparison_df.to_csv('models/model_comparison.csv')
    joblib.dump(all_predictions, 'models/all_predictions.pkl')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nSaved files:")
    print("  • models/lstm_model.keras")
    print("  • models/gru_model.keras")
    print("  • models/transformer_model.keras")
    print("  • models/xgboost_model.pkl")
    print("  • models/model_comparison.csv")
    print("  • models/all_predictions.pkl")
    print("  • models/scalers_seq/")
    print("  • models/scalers_ml/")
    print("\nYou can now run predictions: python predict.py")
    print("="*80 + "\n")
    
    return all_results, all_predictions


if __name__ == "__main__":
    results, predictions = train_all_models()
