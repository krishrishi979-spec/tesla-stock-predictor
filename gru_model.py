"""
Model 2: GRU (Gated Recurrent Unit) Neural Network
Best for: Faster training with similar performance to LSTM, better for real-time
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import os


class GRUStockPredictor:
    def __init__(self, sequence_length=60, n_features=None):
        """
        Initialize GRU model
        
        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of input features
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self, gru_units=[128, 64, 32], dropout_rate=0.2, learning_rate=0.001):
        """
        Build GRU architecture
        
        Args:
            gru_units: List of units for each GRU layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        model = Sequential(name='GRU_Stock_Predictor')
        
        # First GRU layer
        model.add(GRU(
            units=gru_units[0],
            return_sequences=True,
            input_shape=(self.sequence_length, self.n_features),
            name='GRU_1'
        ))
        model.add(BatchNormalization(name='BN_1'))
        model.add(Dropout(dropout_rate, name='Dropout_1'))
        
        # Second GRU layer
        model.add(GRU(
            units=gru_units[1],
            return_sequences=True,
            name='GRU_2'
        ))
        model.add(BatchNormalization(name='BN_2'))
        model.add(Dropout(dropout_rate, name='Dropout_2'))
        
        # Third GRU layer
        model.add(GRU(
            units=gru_units[2],
            return_sequences=False,
            name='GRU_3'
        ))
        model.add(BatchNormalization(name='BN_3'))
        model.add(Dropout(dropout_rate, name='Dropout_3'))
        
        # Dense layers
        model.add(Dense(32, activation='relu', name='Dense_1'))
        model.add(Dropout(dropout_rate / 2, name='Dropout_4'))
        
        model.add(Dense(16, activation='relu', name='Dense_2'))
        
        # Output layer
        model.add(Dense(1, name='Output'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        
        print("\n" + "="*60)
        print("GRU Model Architecture")
        print("="*60)
        model.summary()
        print("="*60 + "\n")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train the GRU model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/gru_best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        print("Training GRU model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}\n")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n✓ GRU training completed!")
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def save_model(self, filepath='models/gru_model.keras'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
        # Save training history
        if self.history is not None:
            history_path = filepath.replace('.keras', '_history.pkl')
            joblib.dump(self.history.history, history_path)
        
        print(f"✓ GRU model saved to {filepath}")
    
    def load_model(self, filepath='models/gru_model.keras'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        
        # Load training history if available
        history_path = filepath.replace('.keras', '_history.pkl')
        if os.path.exists(history_path):
            self.history = type('obj', (object,), {'history': joblib.load(history_path)})()
        
        print(f"✓ GRU model loaded from {filepath}")
    
    def get_training_history(self):
        """Get training history for visualization"""
        if self.history is None:
            return None
        return self.history.history


# Example usage
if __name__ == "__main__":
    print("GRU Stock Predictor Module")
    print("This module provides GRU neural network for stock prediction")
    print("\nUsage:")
    print("  from models.gru_model import GRUStockPredictor")
    print("  model = GRUStockPredictor(sequence_length=60, n_features=50)")
    print("  model.build_model()")
    print("  model.train(X_train, y_train, X_val, y_val)")
