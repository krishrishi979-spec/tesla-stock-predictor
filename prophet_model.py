"""
Model 5: Prophet (Facebook/Meta) Time Series Forecasting
Best for: Handling seasonality, trends, and holidays
"""

import numpy as np
import pandas as pd
from prophet import Prophet
import joblib
import os
from datetime import datetime, timedelta


class ProphetStockPredictor:
    def __init__(self):
        """Initialize Prophet model"""
        self.model = None
        self.forecast = None
        
    def build_model(self, changepoint_prior_scale=0.05, 
                    seasonality_prior_scale=10,
                    holidays_prior_scale=10,
                    seasonality_mode='multiplicative',
                    interval_width=0.95):
        """
        Build Prophet model
        
        Args:
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Flexibility of seasonality
            holidays_prior_scale: Flexibility of holiday effects
            seasonality_mode: 'additive' or 'multiplicative'
            interval_width: Width of uncertainty intervals
        """
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
            interval_width=interval_width,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        # Add custom seasonalities
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        print("\n" + "="*60)
        print("Prophet Model Configuration")
        print("="*60)
        print(f"  Changepoint Prior Scale:   {changepoint_prior_scale}")
        print(f"  Seasonality Prior Scale:   {seasonality_prior_scale}")
        print(f"  Holidays Prior Scale:      {holidays_prior_scale}")
        print(f"  Seasonality Mode:          {seasonality_mode}")
        print(f"  Interval Width:            {interval_width}")
        print(f"  Weekly Seasonality:        True")
        print(f"  Yearly Seasonality:        True")
        print(f"  Monthly Seasonality:       True (custom)")
        print("="*60 + "\n")
        
        return self.model
    
    def prepare_data(self, df, target_col='Close', additional_regressors=None):
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns)
        
        Args:
            df: Input dataframe with datetime index
            target_col: Target column name
            additional_regressors: List of additional regressor column names
        
        Returns:
            DataFrame formatted for Prophet
        """
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df[target_col].values
        })
        
        # Add additional regressors
        if additional_regressors:
            for col in additional_regressors:
                if col in df.columns:
                    prophet_df[col] = df[col].values
        
        return prophet_df
    
    def train(self, train_df, additional_regressors=None):
        """
        Train the Prophet model
        
        Args:
            train_df: Training dataframe with 'ds' and 'y' columns
            additional_regressors: List of additional regressor columns
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("Training Prophet model...")
        print(f"Training samples: {len(train_df)}")
        print(f"Date range: {train_df['ds'].min()} to {train_df['ds'].max()}\n")
        
        # Add regressors to model
        if additional_regressors:
            for col in additional_regressors:
                if col in train_df.columns:
                    self.model.add_regressor(col)
                    print(f"  Added regressor: {col}")
        
        # Fit model
        self.model.fit(train_df)
        
        print("\n✓ Prophet training completed!")
        
        return self.model
    
    def predict(self, periods=None, future_df=None):
        """
        Make predictions
        
        Args:
            periods: Number of periods to forecast into the future
            future_df: Custom future dataframe (if not provided, will create one)
        
        Returns:
            Forecast dataframe
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if future_df is None:
            # Create future dataframe
            if periods is None:
                periods = 30  # Default to 30 days
            future_df = self.model.make_future_dataframe(periods=periods)
        
        # Make predictions
        self.forecast = self.model.predict(future_df)
        
        return self.forecast
    
    def predict_for_dates(self, test_df):
        """
        Make predictions for specific dates (for evaluation)
        
        Args:
            test_df: Test dataframe with 'ds' column and optional regressors
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecast = self.model.predict(test_df)
        predictions = forecast['yhat'].values
        
        return predictions
    
    def get_forecast_components(self):
        """
        Get forecast components (trend, seasonality, etc.)
        
        Returns:
            Forecast dataframe with components
        """
        if self.forecast is None:
            raise ValueError("No forecast available. Call predict() first.")
        
        return self.forecast
    
    def plot_forecast(self):
        """Plot the forecast"""
        if self.forecast is None:
            raise ValueError("No forecast available. Call predict() first.")
        
        from prophet.plot import plot_plotly, plot_components_plotly
        
        # Plot forecast
        fig1 = plot_plotly(self.model, self.forecast)
        fig1.show()
        
        # Plot components
        fig2 = plot_components_plotly(self.model, self.forecast)
        fig2.show()
    
    def save_model(self, filepath='models/prophet_model.pkl'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model using joblib
        with open(filepath, 'wb') as f:
            joblib.dump(self.model, f)
        
        # Save forecast if available
        if self.forecast is not None:
            forecast_path = filepath.replace('.pkl', '_forecast.csv')
            self.forecast.to_csv(forecast_path, index=False)
        
        print(f"✓ Prophet model saved to {filepath}")
    
    def load_model(self, filepath='models/prophet_model.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            self.model = joblib.load(f)
        
        # Load forecast if available
        forecast_path = filepath.replace('.pkl', '_forecast.csv')
        if os.path.exists(forecast_path):
            self.forecast = pd.read_csv(forecast_path)
            self.forecast['ds'] = pd.to_datetime(self.forecast['ds'])
        
        print(f"✓ Prophet model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("Prophet Stock Predictor Module")
    print("This module provides Facebook Prophet for stock prediction")
    print("\nUsage:")
    print("  from models.prophet_model import ProphetStockPredictor")
    print("  model = ProphetStockPredictor()")
    print("  model.build_model()")
    print("  train_df = model.prepare_data(data, target_col='Close')")
    print("  model.train(train_df)")
    print("  forecast = model.predict(periods=30)")
