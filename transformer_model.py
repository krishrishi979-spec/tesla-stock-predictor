"""
Model 3: Transformer Neural Network for Time Series
Best for: Capturing complex patterns with attention mechanisms
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import os


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerStockPredictor:
    def __init__(self, sequence_length=60, n_features=None):
        """
        Initialize Transformer model
        
        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of input features
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self, head_size=256, num_heads=4, ff_dim=4, 
                    num_transformer_blocks=3, mlp_units=[128], 
                    dropout=0.2, learning_rate=0.001):
        """
        Build Transformer architecture
        
        Args:
            head_size: Size of attention heads
            num_heads: Number of attention heads
            ff_dim: Hidden layer size in feed forward network inside transformer
            num_transformer_blocks: Number of transformer blocks
            mlp_units: List of dense layer units after transformer blocks
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))
        x = inputs
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            x = TransformerBlock(head_size, num_heads, head_size * ff_dim, dropout)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        
        # MLP layers
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(dropout)(x)
        
        # Output layer
        outputs = layers.Dense(1)(x)
        
        model = Model(inputs, outputs, name='Transformer_Stock_Predictor')
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        
        print("\n" + "="*60)
        print("Transformer Model Architecture")
        print("="*60)
        model.summary()
        print("="*60 + "\n")
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train the Transformer model
        
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
                'models/transformer_best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        print("Training Transformer model...")
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
        
        print("\n✓ Transformer training completed!")
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def save_model(self, filepath='models/transformer_model.keras'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
        # Save training history
        if self.history is not None:
            history_path = filepath.replace('.keras', '_history.pkl')
            joblib.dump(self.history.history, history_path)
        
        print(f"✓ Transformer model saved to {filepath}")
    
    def load_model(self, filepath='models/transformer_model.keras'):
        """Load a trained model"""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'TransformerBlock': TransformerBlock}
        )
        
        # Load training history if available
        history_path = filepath.replace('.keras', '_history.pkl')
        if os.path.exists(history_path):
            self.history = type('obj', (object,), {'history': joblib.load(history_path)})()
        
        print(f"✓ Transformer model loaded from {filepath}")
    
    def get_training_history(self):
        """Get training history for visualization"""
        if self.history is None:
            return None
        return self.history.history


# Example usage
if __name__ == "__main__":
    print("Transformer Stock Predictor Module")
    print("This module provides Transformer neural network for stock prediction")
    print("\nUsage:")
    print("  from models.transformer_model import TransformerStockPredictor")
    print("  model = TransformerStockPredictor(sequence_length=60, n_features=50)")
    print("  model.build_model()")
    print("  model.train(X_train, y_train, X_val, y_val)")
