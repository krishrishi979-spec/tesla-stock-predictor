"""
Tesla Stock Prediction Script
Make predictions using trained models
SELF-CONTAINED - No external module imports needed
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def load_latest_data():
    """Load the latest processed data"""
    data_path = 'data/tesla_spx500_complete.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        return df
    else:
        print("❌ Data not found! Please run 'python collect_data.py' first.")
        return None


def predict_next_days(n_days=1):
    """
    Predict stock prices using all trained models
    
    Args:
        n_days: Number of days to predict (1-30)
    """
    print("\n" + "="*70)
    print(f"  PREDICTING TESLA STOCK PRICE FOR NEXT {n_days} DAY(S)")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    df = load_latest_data()
    if df is None:
        return
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Last date in data: {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Last closing price: ${df['Close'].iloc[-1]:.2f}\n")
    
    # Generate future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=n_days, freq='D')
    
    predictions = {}
    
    # =================================================================
    # Check for trained models
    # =================================================================
    print("Checking for trained models...")
    
    models_found = []
    if os.path.exists('models/lstm_model.keras'):
        models_found.append('LSTM')
    if os.path.exists('models/gru_model.keras'):
        models_found.append('GRU')
    if os.path.exists('models/transformer_model.keras'):
        models_found.append('Transformer')
    if os.path.exists('models/xgboost_model.pkl'):
        models_found.append('XGBoost')
    
    if len(models_found) == 0:
        print("❌ No trained models found!")
        print("   Please run 'python train_models.py' first to train the models.")
        return
    
    print(f"✓ Found trained models: {', '.join(models_found)}\n")
    
    # =================================================================
    # Load scalers
    # =================================================================
    scaler_X = None
    scaler_y = None
    feature_columns = None
    
    try:
        scaler_X = joblib.load('models/scalers_seq/scaler_X.pkl')
        scaler_y = joblib.load('models/scalers_seq/scaler_y.pkl')
        feature_columns = joblib.load('models/scalers_seq/feature_columns.pkl')
        print("✓ Loaded sequence model scalers")
    except Exception as e:
        print(f"⚠ Warning: Could not load sequence scalers: {e}")
    
    # =================================================================
    # Load TensorFlow models
    # =================================================================
    if scaler_X is not None and scaler_y is not None:
        try:
            # Import TensorFlow here to avoid import errors if not installed
            from tensorflow.keras.models import load_model
            
            # Prepare input data - last 60 days
            last_60_days = df.iloc[-60:].copy()
            X_last = last_60_days[feature_columns].values
            X_scaled = scaler_X.transform(X_last)
            X_seq = X_scaled.reshape(1, 60, -1)
            
            # LSTM
            if 'LSTM' in models_found:
                try:
                    print("Loading LSTM model...")
                    lstm_model = load_model('models/lstm_model.keras')
                    pred_scaled = lstm_model.predict(X_seq, verbose=0)
                    pred_price = scaler_y.inverse_transform(pred_scaled)[0][0]
                    predictions['LSTM'] = float(pred_price)
                    print(f"✓ LSTM prediction: ${pred_price:.2f}")
                except Exception as e:
                    print(f"✗ LSTM prediction failed: {e}")
            
            # GRU
            if 'GRU' in models_found:
                try:
                    print("Loading GRU model...")
                    gru_model = load_model('models/gru_model.keras')
                    pred_scaled = gru_model.predict(X_seq, verbose=0)
                    pred_price = scaler_y.inverse_transform(pred_scaled)[0][0]
                    predictions['GRU'] = float(pred_price)
                    print(f"✓ GRU prediction: ${pred_price:.2f}")
                except Exception as e:
                    print(f"✗ GRU prediction failed: {e}")
            
            # Transformer
            if 'Transformer' in models_found:
                try:
                    print("Loading Transformer model...")
                    transformer_model = load_model('models/transformer_model.keras')
                    pred_scaled = transformer_model.predict(X_seq, verbose=0)
                    pred_price = scaler_y.inverse_transform(pred_scaled)[0][0]
                    predictions['Transformer'] = float(pred_price)
                    print(f"✓ Transformer prediction: ${pred_price:.2f}")
                except Exception as e:
                    print(f"✗ Transformer prediction failed: {e}")
                    
        except ImportError:
            print("⚠ TensorFlow not available. Skipping deep learning models.")
        except Exception as e:
            print(f"⚠ Error loading TensorFlow models: {e}")
    
    # =================================================================
    # XGBoost Predictions
    # =================================================================
    if 'XGBoost' in models_found:
        try:
            print("Loading XGBoost model...")
            xgb_model = joblib.load('models/xgboost_model.pkl')
            
            # Load ML scaler
            scaler_X_ml = joblib.load('models/scalers_ml/scaler_X.pkl')
            feature_columns_ml = joblib.load('models/scalers_ml/feature_columns.pkl')
            
            # Prepare features - use last row
            X_last_ml = df[feature_columns_ml].iloc[-1:].values
            X_scaled_ml = scaler_X_ml.transform(X_last_ml)
            
            # Predict
            pred_price = xgb_model.predict(X_scaled_ml)[0]
            predictions['XGBoost'] = float(pred_price)
            print(f"✓ XGBoost prediction: ${pred_price:.2f}")
            
        except Exception as e:
            print(f"✗ XGBoost prediction failed: {e}")
    
    # =================================================================
    # Display Results
    # =================================================================
    print("\n" + "="*70)
    print("  PREDICTION SUMMARY")
    print("="*70 + "\n")
    
    if len(predictions) == 0:
        print("❌ No predictions could be made.")
        print("   Please check that models are trained correctly.")
        return
    
    current_price = float(df['Close'].iloc[-1])
    print(f"Current Price (Last Close): ${current_price:.2f}")
    print(f"Prediction Date: {future_dates[0].strftime('%Y-%m-%d')}\n")
    
    # Display all predictions in a table
    print(f"{'Model':<15} {'Prediction':>12} {'Signal':>6}  {'Change':>20}")
    print("-" * 58)
    
    for model_name, pred_price in predictions.items():
        change = pred_price - current_price
        change_pct = (change / current_price) * 100
        direction = "📈" if change > 0 else "📉"
        print(f"{model_name:<15} ${pred_price:>10.2f}  {direction}     ({change:+.2f}, {change_pct:+.2f}%)")
    
    # Calculate average prediction
    avg_pred = np.mean(list(predictions.values()))
    avg_change = avg_pred - current_price
    avg_change_pct = (avg_change / current_price) * 100
    
    print("-" * 58)
    avg_direction = "📈" if avg_change > 0 else "📉"
    print(f"{'AVERAGE':<15} ${avg_pred:>10.2f}  {avg_direction}     ({avg_change:+.2f}, {avg_change_pct:+.2f}%)")
    
    # Consensus analysis
    print("\n" + "-"*70)
    bullish = sum(1 for p in predictions.values() if p > current_price)
    bearish = len(predictions) - bullish
    
    print(f"Consensus: {bullish} Bullish 📈 | {bearish} Bearish 📉")
    
    # Overall sentiment
    if avg_change_pct > 2:
        sentiment = "🟢 STRONG BUY SIGNAL"
    elif avg_change_pct > 0:
        sentiment = "🟢 BUY SIGNAL"
    elif avg_change_pct > -2:
        sentiment = "🟡 NEUTRAL / HOLD"
    else:
        sentiment = "🔴 SELL SIGNAL"
    
    print(f"Overall Sentiment: {sentiment}")
    
    # Disclaimer
    print("\n" + "="*70)
    print("\n⚠️  IMPORTANT DISCLAIMER:")
    print("   These predictions are for EDUCATIONAL PURPOSES ONLY.")
    print("   Do NOT use for actual trading without professional advice!")
    print("   Stock prices are influenced by many unpredictable factors.")
    print("   Past performance does not guarantee future results.")
    print("\n" + "="*70 + "\n")
    
    return predictions


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Predict Tesla stock prices using trained AI models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py              # Predict next day
  python predict.py --days 7     # Predict next 7 days
  python predict.py --days 30    # Predict next 30 days
        """
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=1,
        help='Number of days to predict (default: 1, max: 30)'
    )
    
    args = parser.parse_args()
    
    if args.days < 1 or args.days > 30:
        print("❌ Error: Days must be between 1 and 30")
        return
    
    predict_next_days(args.days)


if __name__ == "__main__":
    main()
