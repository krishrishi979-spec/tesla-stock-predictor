"""
Tesla Stock Prediction - Standalone Application
This file is designed to be compiled into an executable

To create executable:
    pip install pyinstaller
    pyinstaller --onefile --name=TeslaPredictor --console tesla_predictor_app.py

Then copy the data/ and models/ folders next to the .exe file
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# Get the directory where the executable/script is located
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    APP_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Change to app directory
os.chdir(APP_DIR)

# Paths
DATA_PATH = os.path.join(APP_DIR, 'data', 'tesla_spx500_complete.csv')
MODELS_DIR = os.path.join(APP_DIR, 'models')


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Print application banner"""
    print("\n" + "="*70)
    print("""
    ████████╗███████╗███████╗██╗      █████╗ 
    ╚══██╔══╝██╔════╝██╔════╝██║     ██╔══██╗
       ██║   █████╗  ███████╗██║     ███████║
       ██║   ██╔══╝  ╚════██║██║     ██╔══██║
       ██║   ███████╗███████║███████╗██║  ██║
       ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝
                                              
    📈 STOCK PRICE PREDICTOR 📈
    AI-Powered Forecasting System
    """)
    print("="*70)


def load_data():
    """Load the stock data"""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        return df
    return None


def get_available_models():
    """Check which models are available"""
    models = {}
    
    model_files = {
        'LSTM': 'lstm_model.keras',
        'GRU': 'gru_model.keras',
        'Transformer': 'transformer_model.keras',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = path
    
    return models


def load_scalers():
    """Load the data scalers"""
    scalers = {}
    
    try:
        scalers['seq_X'] = joblib.load(os.path.join(MODELS_DIR, 'scalers_seq', 'scaler_X.pkl'))
        scalers['seq_y'] = joblib.load(os.path.join(MODELS_DIR, 'scalers_seq', 'scaler_y.pkl'))
        scalers['seq_features'] = joblib.load(os.path.join(MODELS_DIR, 'scalers_seq', 'feature_columns.pkl'))
    except:
        pass
    
    try:
        scalers['ml_X'] = joblib.load(os.path.join(MODELS_DIR, 'scalers_ml', 'scaler_X.pkl'))
        scalers['ml_features'] = joblib.load(os.path.join(MODELS_DIR, 'scalers_ml', 'feature_columns.pkl'))
    except:
        pass
    
    return scalers


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def make_predictions():
    """Make predictions using all available models"""
    print("\n" + "="*70)
    print("  🔮 MAKING PREDICTIONS...")
    print("="*70 + "\n")
    
    # Load data
    df = load_data()
    if df is None:
        print("❌ ERROR: Data file not found!")
        print(f"   Expected at: {DATA_PATH}")
        print("\n   Please ensure 'data/tesla_spx500_complete.csv' exists.")
        return None
    
    current_price = float(df['Close'].iloc[-1])
    last_date = df.index[-1]
    
    print(f"📊 Data loaded: {len(df)} records")
    print(f"📅 Last date: {last_date.strftime('%Y-%m-%d')}")
    print(f"💰 Current price: ${current_price:.2f}\n")
    
    # Get available models
    models = get_available_models()
    if not models:
        print("❌ ERROR: No trained models found!")
        print(f"   Expected in: {MODELS_DIR}")
        return None
    
    print(f"🤖 Available models: {', '.join(models.keys())}\n")
    
    # Load scalers
    scalers = load_scalers()
    
    predictions = {}
    
    # Deep Learning Models (LSTM, GRU, Transformer)
    if 'seq_X' in scalers and 'seq_y' in scalers:
        try:
            from tensorflow.keras.models import load_model
            
            # Prepare sequence data
            features = scalers['seq_features']
            last_60 = df.iloc[-60:][features].values
            X_scaled = scalers['seq_X'].transform(last_60)
            X_seq = X_scaled.reshape(1, 60, -1)
            
            for model_name in ['LSTM', 'GRU', 'Transformer']:
                if model_name in models:
                    try:
                        print(f"   Loading {model_name}...", end=" ")
                        model = load_model(models[model_name])
                        pred = model.predict(X_seq, verbose=0)
                        price = float(scalers['seq_y'].inverse_transform(pred)[0][0])
                        predictions[model_name] = price
                        print(f"✓ ${price:.2f}")
                    except Exception as e:
                        print(f"✗ Failed: {e}")
                        
        except ImportError:
            print("⚠️  TensorFlow not available - skipping deep learning models")
    
    # XGBoost
    if 'XGBoost' in models and 'ml_X' in scalers:
        try:
            print(f"   Loading XGBoost...", end=" ")
            xgb_model = joblib.load(models['XGBoost'])
            features = scalers['ml_features']
            X_ml = df[features].iloc[-1:].values
            X_scaled = scalers['ml_X'].transform(X_ml)
            price = float(xgb_model.predict(X_scaled)[0])
            predictions['XGBoost'] = price
            print(f"✓ ${price:.2f}")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    return predictions, current_price, last_date


def display_results(predictions, current_price, last_date):
    """Display prediction results"""
    if not predictions:
        print("\n❌ No predictions available!")
        return
    
    next_date = last_date + timedelta(days=1)
    
    print("\n" + "="*70)
    print("  📈 PREDICTION RESULTS")
    print("="*70)
    
    print(f"\n  Current Price: ${current_price:.2f}")
    print(f"  Prediction For: {next_date.strftime('%Y-%m-%d')}")
    
    print("\n  " + "-"*60)
    print(f"  {'MODEL':<15} {'PREDICTION':>12} {'CHANGE':>12} {'SIGNAL':>10}")
    print("  " + "-"*60)
    
    for model, price in predictions.items():
        change = price - current_price
        pct = (change / current_price) * 100
        
        if change > 0:
            signal = "📈 BUY"
            color = "+"
        else:
            signal = "📉 SELL"
            color = ""
        
        print(f"  {model:<15} ${price:>10.2f} {color}{change:>+10.2f} ({pct:>+.2f}%)  {signal}")
    
    # Average
    avg_price = np.mean(list(predictions.values()))
    avg_change = avg_price - current_price
    avg_pct = (avg_change / current_price) * 100
    
    print("  " + "-"*60)
    print(f"  {'AVERAGE':<15} ${avg_price:>10.2f} {avg_change:>+10.2f} ({avg_pct:>+.2f}%)")
    print("  " + "-"*60)
    
    # Consensus
    bullish = sum(1 for p in predictions.values() if p > current_price)
    bearish = len(predictions) - bullish
    
    print(f"\n  📊 CONSENSUS: {bullish} Bullish 📈 | {bearish} Bearish 📉")
    
    # Overall sentiment
    if avg_pct > 3:
        sentiment = "🟢🟢 STRONG BUY"
    elif avg_pct > 1:
        sentiment = "🟢 BUY"
    elif avg_pct > -1:
        sentiment = "🟡 HOLD"
    elif avg_pct > -3:
        sentiment = "🔴 SELL"
    else:
        sentiment = "🔴🔴 STRONG SELL"
    
    print(f"  📊 SENTIMENT: {sentiment}")
    
    print("\n" + "="*70)


def show_disclaimer():
    """Show disclaimer"""
    print("\n" + "="*70)
    print("  ⚠️  IMPORTANT DISCLAIMER")
    print("="*70)
    print("""
  This software is for EDUCATIONAL PURPOSES ONLY.
  
  • Do NOT use for actual trading without professional advice
  • Stock predictions are inherently uncertain
  • Past performance does not guarantee future results
  • Always consult a financial advisor before investing
  • The creators are not responsible for any financial losses
    """)
    print("="*70)


def show_menu():
    """Show main menu"""
    print("\n" + "-"*40)
    print("  MENU:")
    print("-"*40)
    print("  1. Make Prediction")
    print("  2. View Data Summary")
    print("  3. View Model Info")
    print("  4. Show Disclaimer")
    print("  5. Exit")
    print("-"*40)


def view_data_summary():
    """View data summary"""
    print("\n" + "="*70)
    print("  📊 DATA SUMMARY")
    print("="*70 + "\n")
    
    df = load_data()
    if df is None:
        print("❌ Data not found!")
        return
    
    print(f"  Date Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total Records: {len(df)}")
    print(f"  Total Features: {len(df.columns)}")
    
    print("\n  Latest 5 Days:")
    print("  " + "-"*60)
    latest = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5)
    for idx, row in latest.iterrows():
        print(f"  {idx.strftime('%Y-%m-%d')}  O: ${row['Open']:.2f}  H: ${row['High']:.2f}  "
              f"L: ${row['Low']:.2f}  C: ${row['Close']:.2f}")
    
    print("\n  Price Statistics:")
    print(f"    Min: ${df['Close'].min():.2f}")
    print(f"    Max: ${df['Close'].max():.2f}")
    print(f"    Mean: ${df['Close'].mean():.2f}")


def view_model_info():
    """View model information"""
    print("\n" + "="*70)
    print("  🤖 MODEL INFORMATION")
    print("="*70 + "\n")
    
    models = get_available_models()
    
    if not models:
        print("❌ No models found!")
        return
    
    print("  Available Models:")
    print("  " + "-"*50)
    
    for name, path in models.items():
        size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"  ✓ {name:<15} ({size:.2f} MB)")
    
    # Check for comparison file
    comp_path = os.path.join(MODELS_DIR, 'model_comparison.csv')
    if os.path.exists(comp_path):
        print("\n  Performance Metrics:")
        print("  " + "-"*50)
        comp = pd.read_csv(comp_path, index_col=0)
        for idx, row in comp.iterrows():
            print(f"  {idx}: RMSE=${row['RMSE']:.2f}, Accuracy={row['Directional_Accuracy']:.1f}%")


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point"""
    clear_screen()
    print_banner()
    
    # Check initial status
    df = load_data()
    models = get_available_models()
    
    print("\n  📋 STATUS CHECK:")
    print(f"    Data: {'✓ Loaded (' + str(len(df)) + ' records)' if df is not None else '❌ Not found'}")
    print(f"    Models: {', '.join(models.keys()) if models else '❌ None found'}")
    
    if df is None or not models:
        print("\n  ⚠️  Make sure 'data/' and 'models/' folders are in the same")
        print("     directory as this executable!")
    
    # Main loop
    while True:
        show_menu()
        
        try:
            choice = input("\n  Enter choice (1-5): ").strip()
            
            if choice == '1':
                result = make_predictions()
                if result:
                    predictions, current_price, last_date = result
                    display_results(predictions, current_price, last_date)
                    
            elif choice == '2':
                view_data_summary()
                
            elif choice == '3':
                view_model_info()
                
            elif choice == '4':
                show_disclaimer()
                
            elif choice == '5':
                print("\n  👋 Goodbye! Thank you for using Tesla Stock Predictor!")
                print("="*70 + "\n")
                break
                
            else:
                print("\n  ❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n  👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
        
        input("\n  Press Enter to continue...")
        clear_screen()
        print_banner()


if __name__ == "__main__":
    main()
