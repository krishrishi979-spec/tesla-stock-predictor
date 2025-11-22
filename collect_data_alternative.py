"""
Alternative Data Collection using Alpha Vantage API
FREE: 25 requests per day (enough for this project)

Get your FREE API key at: https://www.alphavantage.co/support/#api-key
"""

import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
import time


def download_from_alpha_vantage(symbol, api_key, save_path='data/'):
    """
    Download stock data from Alpha Vantage (FREE)
    
    Get FREE API key at: https://www.alphavantage.co/support/#api-key
    """
    print(f"Downloading {symbol} from Alpha Vantage...")
    
    # Alpha Vantage API endpoint for daily data
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': 'full',  # Get all available data
        'apikey': api_key,
        'datatype': 'csv'
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            # Save raw CSV
            os.makedirs(save_path, exist_ok=True)
            filepath = f"{save_path}{symbol}_raw.csv"
            
            with open(filepath, 'w') as f:
                f.write(response.text)
            
            # Load and process
            df = pd.read_csv(filepath)
            
            if 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'Date'})
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                df = df.sort_index()
                
                # Rename columns to standard names
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                print(f"✓ Downloaded {len(df)} records for {symbol}")
                df.to_csv(filepath)
                return df
            else:
                print(f"Error: Unexpected response format")
                print(f"Response: {response.text[:500]}")
                return None
        else:
            print(f"Error: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
        return None


def download_from_stooq(symbol, save_path='data/'):
    """
    Download stock data from Stooq (FREE, no API key needed)
    Works for US stocks and indices
    """
    print(f"Downloading {symbol} from Stooq...")
    
    # Stooq uses different symbol format
    stooq_symbol = symbol.lower()
    if symbol == '^GSPC':
        stooq_symbol = '^spx'
    
    url = f'https://stooq.com/q/d/l/?s={stooq_symbol}&i=d'
    
    try:
        df = pd.read_csv(url)
        
        if len(df) > 0:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            df = df.sort_index()
            
            os.makedirs(save_path, exist_ok=True)
            clean_symbol = symbol.replace('^', '')
            filepath = f"{save_path}{clean_symbol}_raw.csv"
            df.to_csv(filepath)
            
            print(f"✓ Downloaded {len(df)} records for {symbol}")
            return df
        else:
            print(f"No data received for {symbol}")
            return None
            
    except Exception as e:
        print(f"Error downloading {symbol} from Stooq: {e}")
        return None


def download_from_nasdaq(save_path='data/'):
    """
    Download Tesla data directly from NASDAQ (FREE)
    Note: Only works for NASDAQ-listed stocks
    """
    print("Downloading TSLA from NASDAQ...")
    
    url = "https://api.nasdaq.com/api/quote/TSLA/historical"
    params = {
        'assetclass': 'stocks',
        'fromdate': '2015-01-01',
        'limit': 9999,
        'todate': datetime.now().strftime('%Y-%m-%d')
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data and 'tradesTable' in data['data']:
                rows = data['data']['tradesTable']['rows']
                df = pd.DataFrame(rows)
                
                # Process columns
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date')
                df = df.sort_index()
                
                # Rename and convert columns
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Convert string values to float
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
                df['Volume'] = df['Volume'].str.replace(',', '').astype(float)
                
                os.makedirs(save_path, exist_ok=True)
                df.to_csv(f"{save_path}TSLA_raw.csv")
                
                print(f"✓ Downloaded {len(df)} records for TSLA")
                return df
        
        print("Failed to download from NASDAQ")
        return None
        
    except Exception as e:
        print(f"Error downloading from NASDAQ: {e}")
        return None


def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    print("Adding technical indicators...")
    
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Returns
    df['Returns'] = close.pct_change()
    df['Log_Returns'] = np.log(close / close.shift(1))
    
    # Moving Averages
    for window in [5, 10, 20, 50, 200]:
        df[f'SMA_{window}'] = close.rolling(window=window).mean()
    
    df['EMA_12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close.ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
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
    
    # Stochastic
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    df['Stochastic'] = 100 * (close - low_14) / (high_14 - low_14)
    
    # ATR
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Volume indicators
    df['Volume_SMA_20'] = volume.rolling(window=20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_20']
    
    # Momentum
    df['Momentum_5'] = close - close.shift(5)
    df['Momentum_10'] = close - close.shift(10)
    df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100
    
    # OBV
    df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    return df


def main():
    """Main function to collect data using alternative sources"""
    print("\n" + "="*60)
    print("ALTERNATIVE DATA COLLECTION")
    print("="*60)
    print("\nChoose a data source:")
    print("1. Stooq (FREE, no API key needed)")
    print("2. Alpha Vantage (FREE, requires API key)")
    print("3. Manual CSV files (already downloaded)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    os.makedirs('data', exist_ok=True)
    
    if choice == '1':
        # Download from Stooq
        print("\n" + "-"*40)
        print("Downloading from Stooq (FREE)...")
        print("-"*40)
        
        tesla_df = download_from_stooq('TSLA')
        time.sleep(2)  # Be nice to the server
        spx_df = download_from_stooq('^GSPC')
        
        if tesla_df is None or spx_df is None:
            print("\n✗ Stooq download failed. Try option 3 (manual download).")
            return
            
    elif choice == '2':
        # Download from Alpha Vantage
        print("\n" + "-"*40)
        print("Alpha Vantage (FREE API)")
        print("-"*40)
        print("\nGet your FREE API key at:")
        print("https://www.alphavantage.co/support/#api-key")
        
        api_key = input("\nEnter your Alpha Vantage API key: ").strip()
        
        if not api_key:
            print("No API key provided. Exiting.")
            return
        
        tesla_df = download_from_alpha_vantage('TSLA', api_key)
        print("\nWaiting 15 seconds (API rate limit)...")
        time.sleep(15)
        
        # Note: Alpha Vantage doesn't have S&P 500 in free tier
        print("\nNote: S&P 500 not available in Alpha Vantage free tier.")
        print("Using Tesla data only (without S&P 500 correlation).")
        spx_df = None
        
    elif choice == '3':
        # Process manual downloads
        print("\n" + "-"*40)
        print("Processing Manual Downloads")
        print("-"*40)
        print("\nLooking for CSV files in data/ folder...")
        
        # Try to load Tesla data
        tsla_files = ['data/TSLA_raw.csv', 'data/TSLA.csv', 'data/tsla.csv']
        tesla_df = None
        for f in tsla_files:
            if os.path.exists(f):
                print(f"Found: {f}")
                tesla_df = pd.read_csv(f, index_col='Date', parse_dates=True)
                break
        
        if tesla_df is None:
            print("\n✗ Tesla data not found!")
            print("\nPlease download from Yahoo Finance:")
            print("1. Go to: https://finance.yahoo.com/quote/TSLA/history")
            print("2. Set Time Period to 'Max'")
            print("3. Click 'Download' button")
            print("4. Save as 'data/TSLA.csv'")
            return
        
        # Try to load S&P 500 data
        spx_files = ['data/GSPC_raw.csv', 'data/GSPC.csv', 'data/^GSPC.csv', 'data/gspc.csv']
        spx_df = None
        for f in spx_files:
            if os.path.exists(f):
                print(f"Found: {f}")
                spx_df = pd.read_csv(f, index_col='Date', parse_dates=True)
                break
        
        if spx_df is None:
            print("\nNote: S&P 500 data not found. Continuing without it.")
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Process the data
    print("\n" + "-"*40)
    print("Processing Data...")
    print("-"*40)
    
    # Add technical indicators to Tesla
    tesla_df = add_technical_indicators(tesla_df)
    
    # If we have S&P 500 data, merge it
    if spx_df is not None:
        spx_df = add_technical_indicators(spx_df)
        
        # Create SPX features
        spx_features = pd.DataFrame(index=spx_df.index)
        spx_features['SPX_Close'] = spx_df['Close']
        spx_features['SPX_Volume'] = spx_df['Volume']
        spx_features['SPX_Returns'] = spx_df['Returns']
        spx_features['SPX_SMA_20'] = spx_df['SMA_20']
        spx_features['SPX_SMA_50'] = spx_df['SMA_50']
        spx_features['SPX_RSI'] = spx_df['RSI']
        
        # Merge
        tesla_df = tesla_df.join(spx_features, how='left')
        
        # Correlation
        tesla_df['TSLA_SPX_Correlation'] = tesla_df['Returns'].rolling(window=20).corr(tesla_df['SPX_Returns'])
    
    # Drop NaN
    print(f"\nRemoving NaN values...")
    print(f"Before: {len(tesla_df)} rows")
    tesla_df = tesla_df.dropna()
    print(f"After: {len(tesla_df)} rows")
    
    # Save final dataset
    output_file = 'data/tesla_spx500_complete.csv'
    tesla_df.to_csv(output_file)
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)
    print(f"\n✓ Saved to: {output_file}")
    print(f"✓ Date Range: {tesla_df.index[0]} to {tesla_df.index[-1]}")
    print(f"✓ Total Records: {len(tesla_df)}")
    print(f"✓ Total Features: {len(tesla_df.columns)}")
    print("\nNext step: python train_models.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
