"""
Model 4: XGBoost (Extreme Gradient Boosting)
Best for: Feature importance analysis and interpretability
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import os
import pandas as pd


class XGBoostStockPredictor:
    def __init__(self):
        """Initialize XGBoost model"""
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def build_model(self, n_estimators=1000, learning_rate=0.01, 
                    max_depth=7, subsample=0.8, colsample_bytree=0.8,
                    gamma=0, min_child_weight=1):
        """
        Build XGBoost model
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Step size shrinkage
            max_depth: Maximum tree depth
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            gamma: Minimum loss reduction for split
            min_child_weight: Minimum sum of instance weight in child
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight,
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        
        print("\n" + "="*60)
        print("XGBoost Model Configuration")
        print("="*60)
        print(f"  N Estimators:       {n_estimators}")
        print(f"  Learning Rate:      {learning_rate}")
        print(f"  Max Depth:          {max_depth}")
        print(f"  Subsample:          {subsample}")
        print(f"  Colsample by Tree:  {colsample_bytree}")
        print(f"  Gamma:              {gamma}")
        print(f"  Min Child Weight:   {min_child_weight}")
        print("="*60 + "\n")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, verbose=True):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Print training progress
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Training XGBoost model...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}\n")
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=verbose
        )
        
        # Get best iteration
        best_iteration = self.model.best_iteration
        print(f"\n✓ XGBoost training completed!")
        print(f"  Best iteration: {best_iteration}")
        
        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, cv=3):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv: Number of cross-validation folds
        """
        print("Starting hyperparameter tuning...")
        print("This may take several minutes...\n")
        
        param_grid = {
            'n_estimators': [500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 9],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print("\n✓ Hyperparameter tuning completed!")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_feature_importance(self, feature_names=None, top_n=20):
        """
        Get feature importance
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath='models/xgboost_model.pkl'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, filepath)
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = filepath.replace('.pkl', '_importance.pkl')
            joblib.dump(self.feature_importance, importance_path)
        
        # Save best params if available
        if self.best_params is not None:
            params_path = filepath.replace('.pkl', '_params.pkl')
            joblib.dump(self.best_params, params_path)
        
        print(f"✓ XGBoost model saved to {filepath}")
    
    def load_model(self, filepath='models/xgboost_model.pkl'):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        
        # Load feature importance if available
        importance_path = filepath.replace('.pkl', '_importance.pkl')
        if os.path.exists(importance_path):
            self.feature_importance = joblib.load(importance_path)
        
        # Load best params if available
        params_path = filepath.replace('.pkl', '_params.pkl')
        if os.path.exists(params_path):
            self.best_params = joblib.load(params_path)
        
        print(f"✓ XGBoost model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("XGBoost Stock Predictor Module")
    print("This module provides XGBoost for stock prediction")
    print("\nUsage:")
    print("  from models.xgboost_model import XGBoostStockPredictor")
    print("  model = XGBoostStockPredictor()")
    print("  model.build_model()")
    print("  model.train(X_train, y_train, X_val, y_val)")
