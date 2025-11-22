"""
Data Collection Script for Tesla Stock Prediction
Downloads Tesla (TSLA) and S&P 500 (^GSPC) historical data
UPDATED: Better error handling and yfinance compatibility
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

# Try to import yfinance
try:
    import yfinance as yf
    # Disable yfinance progress bar and set a longer timeout
    yf.pdr_override()
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    exit(1)


class StockDataCollector:
    def __init__(self, start_date='2015-01-01', end_date=None):
        """
        Initialize data collector
        
        Args:
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (default: today)
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        
    def download_stock_data(self, ticker, save_path='data/', max_retries=3):
        """Download stock data from Yahoo Finance with retry logic"""
        print(f"Downloading {ticker} data from {self.start_date} to {self.end_date}...")
        
        for attempt in range(max_retries):
            try:
                # Method 1: Using Ticker object (more reliable)
                stock = yf.Ticker(ticker)
                data = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)
                
                if data.empty:
                    # Method 2: Try download function as fallback
                    print(f"  Attempt {attempt + 1}: Trying alternative download method...")
                    data = yf.download(
                        ticker, 
                        start=self.start_date, 
                        end=self.end_date, 
                        progress=False,
                        auto_adjust=True,
                        timeout=30
                    )
                
                if data.empty:
                    if attempt < max_retries - 1:
                        print(f"  Attempt {attempt + 1} failed. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        print(f"Warning: No data found for {ticker} after {max_retries} attempts")
                        return None
                
                # Handle multi-level columns from newer yfinance versions
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Ensure standard column names
                # Sometimes yfinance returns lowercase column names
                data.columns = [col.title() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] else col for col in data.columns]
                
                # Rename 'Adj Close' if present
                if 'Adj Close' not in data.columns and 'Adjclose' in data.columns:
                    data = data.rename(columns={'Adjclose': 'Adj Close'})
                
                print(f"Successfully downloaded {len(data)} records for {ticker}")
                
                # Save raw data
                os.makedirs(save_path, exist_ok=True)
                clean_ticker = ticker.replace('^', '')
                data.to_csv(f"{save_path}{clean_ticker}_raw.csv")
                
                return data
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} error: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print(f"Error downloading {ticker} after {max_retries} attempts: {str(e)}")
                    return None
        
        return None
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        print("Adding technical indicators...")
        
        df = df.copy()
        
        # Ensure columns are not multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Get price and volume as Series (handle potential DataFrame issues)
        def to_series(col):
            if isinstance(col, pd.DataFrame):
                return col.iloc[:, 0]
            return col
        
        close = to_series(df['Close'])
        high = to_series(df['High'])
        low = to_series(df['Low'])
        volume = to_series(df['Volume'])
        
        # Price-based indicators
        df['Returns'] = close.pct_change()
        df['Log_Returns'] = np.log(close / close.shift(1))
        
        # Moving Averages
        df['SMA_5'] = close.rolling(window=5).mean()
        df['SMA_10'] = close.rolling(window=10).mean()
        df['SMA_20'] = close.rolling(window=20).mean()
        df['SMA_50'] = close.rolling(window=50).mean()
        df['SMA_200'] = close.rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
        df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['BB_Middle'] = bb_middle
        df['BB_Upper'] = bb_middle + (bb_std * 2)
        df['BB_Lower'] = bb_middle - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # Stochastic Oscillator
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        df['Stochastic'] = 100 * (close - low_14) / (high_14 - low_14)
        
        # ATR (Average True Range) - Volatility
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Volume indicators
        df['Volume_SMA_20'] = volume.rolling(window=20).mean()
        df['Volume_Ratio'] = volume / df['Volume_SMA_20']
        
        # Price momentum
        df['Momentum_5'] = close - close.shift(5)
        df['Momentum_10'] = close - close.shift(10)
        
        # Rate of Change
        df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        # Count indicators added
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Dividends', 'Stock Splits']
        indicator_count = len([col for col in df.columns if col not in base_cols])
        print(f"Added {indicator_count} technical indicators")
        
        return df
    
    def merge_with_spx500(self, tesla_df, spx_df):
        """Merge Tesla data with SPX500 indicators"""
        print("Merging Tesla data with SPX500 indicators...")
        
        # Ensure columns are not multi-index
        if isinstance(spx_df.columns, pd.MultiIndex):
            spx_df.columns = spx_df.columns.get_level_values(0)
        
        # Helper function
        def to_series(col):
            if isinstance(col, pd.DataFrame):
                return col.iloc[:, 0]
            return col
        
        spx_close = to_series(spx_df['Close'])
        spx_volume = to_series(spx_df['Volume'])
        spx_returns = to_series(spx_df['Returns'])
        
        # Add SPX500 features with prefix
        spx_features = pd.DataFrame(index=spx_df.index)
        spx_features['SPX_Close'] = spx_close
        spx_features['SPX_Volume'] = spx_volume
        spx_features['SPX_Returns'] = spx_returns
        
        # Add SPX technical indicators
        spx_features['SPX_SMA_20'] = spx_close.rolling(window=20).mean()
        spx_features['SPX_SMA_50'] = spx_close.rolling(window=50).mean()
        
        # RSI for SPX
        delta = spx_close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        spx_features['SPX_RSI'] = 100 - (100 / (1 + rs))
        
        # Merge on date index
        merged_df = tesla_df.join(spx_features, how='left')
        
        # Add correlation features
        tesla_returns = to_series(merged_df['Returns'])
        spx_ret = to_series(merged_df['SPX_Returns'])
        merged_df['TSLA_SPX_Correlation'] = tesla_returns.rolling(window=20).corr(spx_ret)
        
        print(f"Final dataset shape: {merged_df.shape}")
        
        return merged_df
    
    def prepare_dataset(self, save_path='data/'):
        """Complete pipeline to prepare the dataset"""
        print("\n" + "="*60)
        print("Starting Data Collection and Preparation")
        print("="*60 + "\n")
        
        # Download Tesla data
        tesla_data = self.download_stock_data('TSLA', save_path)
        if tesla_data is None:
            return None
        
        # Download SPX500 data
        spx_data = self.download_stock_data('^GSPC', save_path)
        if spx_data is None:
            return None
        
        # Add technical indicators
        tesla_data = self.add_technical_indicators(tesla_data)
        spx_data = self.add_technical_indicators(spx_data)
        
        # Merge datasets
        final_data = self.merge_with_spx500(tesla_data, spx_data)
        
        # Drop rows with NaN values
        print(f"\nRemoving NaN values...")
        print(f"Before: {len(final_data)} rows")
        final_data = final_data.dropna()
        print(f"After: {len(final_data)} rows")
        
        # Remove any extra columns that might cause issues
        cols_to_remove = ['Dividends', 'Stock Splits', 'Capital Gains']
        for col in cols_to_remove:
            if col in final_data.columns:
                final_data = final_data.drop(columns=[col])
        
        # Save final dataset
        output_file = f"{save_path}tesla_spx500_complete.csv"
        final_data.to_csv(output_file)
        print(f"\n✓ Final dataset saved to: {output_file}")
        
        # Print dataset summary
        print("\n" + "="*60)
        print("Dataset Summary")
        print("="*60)
        print(f"Date Range: {final_data.index[0]} to {final_data.index[-1]}")
        print(f"Total Trading Days: {len(final_data)}")
        print(f"Total Features: {len(final_data.columns)}")
        print(f"\nColumns: {final_data.columns.tolist()}")
        print("="*60 + "\n")
        
        return final_data


def test_yfinance_connection():
    """Test if yfinance can connect to Yahoo Finance"""
    print("Testing yfinance connection...")
    try:
        test_ticker = yf.Ticker("AAPL")
        test_data = test_ticker.history(period="5d")
        if not test_data.empty:
            print("✓ yfinance connection successful!")
            return True
        else:
            print("✗ yfinance returned empty data")
            return False
    except Exception as e:
        print(f"✗ yfinance connection failed: {e}")
        return False


if __name__ == "__main__":
    # First test the connection
    if not test_yfinance_connection():
        print("\n" + "="*60)
        print("TROUBLESHOOTING:")
        print("="*60)
        print("1. Check your internet connection")
        print("2. Try upgrading yfinance: pip install yfinance --upgrade")
        print("3. Yahoo Finance might be temporarily unavailable")
        print("4. Try using a VPN if you're in a restricted region")
        print("5. Wait a few minutes and try again")
        print("="*60 + "\n")
        
        # Ask user if they want to continue anyway
        user_input = input("Do you want to try downloading Tesla data anyway? (y/n): ")
        if user_input.lower() != 'y':
            exit(1)
    
    # Initialize collector
    collector = StockDataCollector(start_date='2015-01-01')
    
    # Prepare complete dataset
    data = collector.prepare_dataset(save_path='data/')
    
    if data is not None:
        print("✓ Data collection completed successfully!")
        print(f"\nFirst few rows:")
        print(data.head())
    else:
        print("✗ Data collection failed!")
        print("\nAlternative: You can manually download data from Yahoo Finance:")
        print("1. Go to https://finance.yahoo.com/quote/TSLA/history")
        print("2. Set date range: Jan 1, 2015 to today")
        print("3. Click 'Download' button")
        print("4. Save as 'data/TSLA_raw.csv'")
        print("5. Repeat for ^GSPC (S&P 500)")
