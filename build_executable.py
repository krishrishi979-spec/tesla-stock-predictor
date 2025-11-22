"""
Build Script - Create Executable for Tesla Stock Prediction
Run this script to create a standalone .exe file

Requirements:
    pip install pyinstaller

Usage:
    python build_executable.py
"""

import os
import sys
import shutil
import subprocess


def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        print(f"✓ PyInstaller version: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("❌ PyInstaller not installed!")
        print("\nInstalling PyInstaller...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
        return True


def create_main_app():
    """Create a main application file that combines all functionality"""
    
    main_app_code = '''"""
Tesla Stock Prediction - Main Application
Standalone executable version
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Get the directory where the executable is located
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    APP_DIR = os.path.dirname(sys.executable)
else:
    # Running as script
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(APP_DIR)


def print_header():
    """Print application header"""
    print("\\n" + "="*70)
    print("  🚀 TESLA STOCK PREDICTION SYSTEM")
    print("  AI-Powered Stock Forecasting")
    print("="*70)


def print_menu():
    """Print main menu"""
    print("\\n📋 MAIN MENU:")
    print("-" * 40)
    print("1. Make Prediction (Next Day)")
    print("2. Make Prediction (Custom Days)")
    print("3. View Model Performance")
    print("4. View Latest Data")
    print("5. Update Data (Download Latest)")
    print("6. Retrain Models")
    print("7. Exit")
    print("-" * 40)


def load_data():
    """Load the processed data"""
    data_path = os.path.join(APP_DIR, 'data', 'tesla_spx500_complete.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        return df
    return None


def check_models():
    """Check which models are available"""
    models_dir = os.path.join(APP_DIR, 'models')
    models = []
    
    if os.path.exists(os.path.join(models_dir, 'lstm_model.keras')):
        models.append('LSTM')
    if os.path.exists(os.path.join(models_dir, 'gru_model.keras')):
        models.append('GRU')
    if os.path.exists(os.path.join(models_dir, 'transformer_model.keras')):
        models.append('Transformer')
    if os.path.exists(os.path.join(models_dir, 'xgboost_model.pkl')):
        models.append('XGBoost')
    
    return models


def make_prediction(n_days=1):
    """Make stock price predictions"""
    print("\\n" + "="*70)
    print(f"  PREDICTING TESLA STOCK PRICE FOR NEXT {n_days} DAY(S)")
    print("="*70 + "\\n")
    
    # Load data
    df = load_data()
    if df is None:
        print("❌ Data not found! Please update data first (Option 5).")
        return
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Last date: {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Last price: ${df['Close'].iloc[-1]:.2f}\\n")
    
    # Check models
    models_found = check_models()
    if not models_found:
        print("❌ No trained models found! Please train models first (Option 6).")
        return
    
    print(f"✓ Found models: {', '.join(models_found)}\\n")
    
    predictions = {}
    models_dir = os.path.join(APP_DIR, 'models')
    
    # Load scalers
    try:
        scaler_X = joblib.load(os.path.join(models_dir, 'scalers_seq', 'scaler_X.pkl'))
        scaler_y = joblib.load(os.path.join(models_dir, 'scalers_seq', 'scaler_y.pkl'))
        feature_columns = joblib.load(os.path.join(models_dir, 'scalers_seq', 'feature_columns.pkl'))
    except Exception as e:
        print(f"⚠ Could not load scalers: {e}")
        scaler_X = None
    
    # TensorFlow predictions
    if scaler_X is not None:
        try:
            from tensorflow.keras.models import load_model
            
            # Prepare data
            last_60 = df.iloc[-60:][feature_columns].values
            X_scaled = scaler_X.transform(last_60)
            X_seq = X_scaled.reshape(1, 60, -1)
            
            for model_name in ['LSTM', 'GRU', 'Transformer']:
                if model_name in models_found:
                    try:
                        model_file = f'{model_name.lower()}_model.keras'
                        model = load_model(os.path.join(models_dir, model_file))
                        pred = model.predict(X_seq, verbose=0)
                        price = scaler_y.inverse_transform(pred)[0][0]
                        predictions[model_name] = float(price)
                        print(f"✓ {model_name}: ${price:.2f}")
                    except Exception as e:
                        print(f"✗ {model_name} failed: {e}")
        except ImportError:
            print("⚠ TensorFlow not available")
    
    # XGBoost prediction
    if 'XGBoost' in models_found:
        try:
            xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
            scaler_ml = joblib.load(os.path.join(models_dir, 'scalers_ml', 'scaler_X.pkl'))
            features_ml = joblib.load(os.path.join(models_dir, 'scalers_ml', 'feature_columns.pkl'))
            
            X_ml = df[features_ml].iloc[-1:].values
            X_scaled_ml = scaler_ml.transform(X_ml)
            price = xgb_model.predict(X_scaled_ml)[0]
            predictions['XGBoost'] = float(price)
            print(f"✓ XGBoost: ${price:.2f}")
        except Exception as e:
            print(f"✗ XGBoost failed: {e}")
    
    # Display results
    if predictions:
        current = float(df['Close'].iloc[-1])
        
        print("\\n" + "="*70)
        print("  PREDICTION SUMMARY")
        print("="*70 + "\\n")
        
        print(f"Current Price: ${current:.2f}")
        print(f"\\n{'Model':<15} {'Prediction':>12} {'Change':>15}")
        print("-" * 45)
        
        for name, pred in predictions.items():
            change = pred - current
            pct = (change / current) * 100
            signal = "📈" if change > 0 else "📉"
            print(f"{name:<15} ${pred:>10.2f}  {signal} {change:+.2f} ({pct:+.2f}%)")
        
        avg = np.mean(list(predictions.values()))
        avg_change = avg - current
        avg_pct = (avg_change / current) * 100
        
        print("-" * 45)
        print(f"{'AVERAGE':<15} ${avg:>10.2f}     {avg_change:+.2f} ({avg_pct:+.2f}%)")
        
        # Sentiment
        bullish = sum(1 for p in predictions.values() if p > current)
        print(f"\\nConsensus: {bullish} Bullish | {len(predictions)-bullish} Bearish")
        
        if avg_pct > 2:
            print("Sentiment: 🟢 STRONG BUY")
        elif avg_pct > 0:
            print("Sentiment: 🟢 BUY")
        elif avg_pct > -2:
            print("Sentiment: 🟡 HOLD")
        else:
            print("Sentiment: 🔴 SELL")
    
    print("\\n⚠️  DISCLAIMER: For educational purposes only!")


def view_performance():
    """View model performance metrics"""
    print("\\n" + "="*70)
    print("  MODEL PERFORMANCE")
    print("="*70 + "\\n")
    
    comp_file = os.path.join(APP_DIR, 'models', 'model_comparison.csv')
    if os.path.exists(comp_file):
        df = pd.read_csv(comp_file, index_col=0)
        print(df.to_string())
    else:
        print("❌ No performance data found. Train models first.")


def view_data():
    """View latest data summary"""
    print("\\n" + "="*70)
    print("  DATA SUMMARY")
    print("="*70 + "\\n")
    
    df = load_data()
    if df is not None:
        print(f"Date Range: {df.index[0]} to {df.index[-1]}")
        print(f"Total Records: {len(df)}")
        print(f"Total Features: {len(df.columns)}")
        print(f"\\nLatest Prices:")
        print(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(5))
    else:
        print("❌ No data found.")


def main():
    """Main application loop"""
    print_header()
    
    # Check initial status
    df = load_data()
    models = check_models()
    
    print("\\n📊 STATUS:")
    print(f"  Data: {'✓ Loaded' if df is not None else '❌ Not found'}")
    print(f"  Models: {', '.join(models) if models else '❌ None found'}")
    
    while True:
        print_menu()
        choice = input("\\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            make_prediction(1)
        elif choice == '2':
            try:
                days = int(input("Enter number of days (1-30): "))
                if 1 <= days <= 30:
                    make_prediction(days)
                else:
                    print("❌ Please enter a number between 1 and 30")
            except ValueError:
                print("❌ Invalid input")
        elif choice == '3':
            view_performance()
        elif choice == '4':
            view_data()
        elif choice == '5':
            print("\\n⚠️  To update data, run: python collect_data.py")
            print("   Or download manually from Yahoo Finance")
        elif choice == '6':
            print("\\n⚠️  To retrain models, run: python train_models.py")
            print("   This requires the full Python environment")
        elif choice == '7':
            print("\\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-7")
        
        input("\\nPress Enter to continue...")


if __name__ == "__main__":
    main()
'''
    
    with open('tesla_predictor_app.py', 'w') as f:
        f.write(main_app_code)
    
    print("✓ Created tesla_predictor_app.py")


def build_executable():
    """Build the executable using PyInstaller"""
    print("\n" + "="*60)
    print("Building Executable...")
    print("="*60 + "\n")
    
    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--onefile',                    # Single executable
        '--name=TeslaStockPredictor',   # Name of executable
        '--console',                     # Console application
        '--clean',                       # Clean build
        '--noconfirm',                  # Don't ask for confirmation
        'tesla_predictor_app.py'        # Main script
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✓ BUILD SUCCESSFUL!")
        print("="*60)
        print("\nExecutable created at: dist/TeslaStockPredictor.exe")
        print("\nTo use the executable:")
        print("1. Copy 'dist/TeslaStockPredictor.exe' to your project folder")
        print("2. Make sure 'data/' and 'models/' folders are in the same location")
        print("3. Double-click to run!")
    else:
        print("\n❌ Build failed!")


def main():
    print("\n" + "="*60)
    print("  TESLA STOCK PREDICTOR - BUILD EXECUTABLE")
    print("="*60)
    
    # Check PyInstaller
    if not check_pyinstaller():
        return
    
    # Create main app file
    print("\nCreating main application file...")
    create_main_app()
    
    # Build
    print("\nBuilding executable (this may take a few minutes)...")
    build_executable()
    
    print("\n" + "="*60)
    print("IMPORTANT: The executable needs these folders to work:")
    print("  - data/          (contains tesla_spx500_complete.csv)")
    print("  - models/        (contains trained model files)")
    print("\nCopy these folders to the same location as the .exe file")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
