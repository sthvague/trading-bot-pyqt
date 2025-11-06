"""
Data Fetcher Module for Trading Bot

This module is responsible for fetching market data for BTCUSD, XAUUSD, and EURUSD
on both H1 (1-hour) and M15 (15-minute) timeframes.

The module uses ccxt for cryptocurrency data and yfinance for forex and gold data.
"""

import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_fetcher')

class DataFetcher:
    """
    Class for fetching and managing market data for the trading bot.
    
    Supports:
    - BTCUSD (Bitcoin)
    - XAUUSD (Gold)
    - EURUSD (Euro/USD)
    
    Timeframes:
    - H1 (1-hour)
    - M15 (15-minute)
    """
    
    def __init__(self, data_dir='./data_storage'):
        """
        Initialize the DataFetcher with storage directory.
        
        Args:
            data_dir (str): Directory to store the fetched data
        """
        self.data_dir = data_dir
        self.timeframes = {
            'H1': '1h',
            'M15': '15m'
        }
        self.symbols = {
            'BTCUSD': {'source': 'ccxt', 'symbol': 'BTC/USDT'},
            'XAUUSD': {'source': 'yfinance', 'symbol': 'GC=F'},  # Gold futures
            'EURUSD': {'source': 'yfinance', 'symbol': 'EURUSD=X'}
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize exchange for cryptocurrency data
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        logger.info("DataFetcher initialized")
    
    def _convert_timeframe_to_yf(self, timeframe):
        """
        Convert internal timeframe to yfinance format.
        
        Args:
            timeframe (str): Internal timeframe format ('H1' or 'M15')
            
        Returns:
            str: yfinance timeframe format
        """
        if timeframe == 'H1':
            return '1h'
        elif timeframe == 'M15':
            return '15m'
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    def _get_filename(self, symbol, timeframe):
        """
        Generate filename for storing data.
        
        Args:
            symbol (str): Market symbol
            timeframe (str): Timeframe
            
        Returns:
            str: Filename
        """
        return os.path.join(self.data_dir, f"{symbol}_{timeframe}.csv")
    
    def fetch_crypto_data(self, symbol, timeframe, limit=1000):
        """
        Fetch cryptocurrency data using ccxt.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe in ccxt format
            limit (int): Number of candles to fetch
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            logger.info(f"Fetching {symbol} data for {timeframe} timeframe")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return None
    
    def fetch_yf_data(self, symbol, timeframe, period='1mo'):
        """
        Fetch data using yfinance.
        
        Args:
            symbol (str): Market symbol in yfinance format
            timeframe (str): Timeframe in yfinance format
            period (str): Period to fetch (e.g., '1mo', '3mo', '6mo')
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            logger.info(f"Fetching {symbol} data for {timeframe} timeframe")
            data = yf.download(
                symbol,
                period=period,
                interval=timeframe,
                progress=False
            )
            
            # Rename columns to match our format
            data.columns = [col.lower() for col in data.columns]
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching yfinance data: {e}")
            return None
    
    def fetch_data(self, symbol_name, timeframe_name, save=True):
        """
        Fetch data for a specific symbol and timeframe.
        
        Args:
            symbol_name (str): Symbol name ('BTCUSD', 'XAUUSD', or 'EURUSD')
            timeframe_name (str): Timeframe name ('H1' or 'M15')
            save (bool): Whether to save the data to file
            
        Returns:
            pd.DataFrame: Market data
        """
        if symbol_name not in self.symbols:
            raise ValueError(f"Unsupported symbol: {symbol_name}")
        
        if timeframe_name not in self.timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe_name}")
        
        symbol_info = self.symbols[symbol_name]
        timeframe = self.timeframes[timeframe_name]
        
        if symbol_info['source'] == 'ccxt':
            df = self.fetch_crypto_data(symbol_info['symbol'], timeframe)
        elif symbol_info['source'] == 'yfinance':
            yf_timeframe = self._convert_timeframe_to_yf(timeframe_name)
            df = self.fetch_yf_data(symbol_info['symbol'], yf_timeframe)
        else:
            raise ValueError(f"Unsupported data source: {symbol_info['source']}")
        
        if df is not None and save:
            filename = self._get_filename(symbol_name, timeframe_name)
            df.to_csv(filename)
            logger.info(f"Data saved to {filename}")
        
        return df
    
    def load_data(self, symbol_name, timeframe_name):
        """
        Load data from file.
        
        Args:
            symbol_name (str): Symbol name
            timeframe_name (str): Timeframe name
            
        Returns:
            pd.DataFrame: Market data
        """
        filename = self._get_filename(symbol_name, timeframe_name)
        
        if os.path.exists(filename):
            logger.info(f"Loading data from {filename}")
            return pd.read_csv(filename, index_col=0, parse_dates=True)
        else:
            logger.warning(f"File {filename} not found. Fetching new data.")
            return self.fetch_data(symbol_name, timeframe_name)
    
    def update_all_data(self):
        """
        Update data for all symbols and timeframes.
        
        Returns:
            dict: Dictionary of dataframes
        """
        data = {}
        
        for symbol_name in self.symbols:
            data[symbol_name] = {}
            for timeframe_name in self.timeframes:
                logger.info(f"Updating {symbol_name} {timeframe_name} data")
                data[symbol_name][timeframe_name] = self.fetch_data(symbol_name, timeframe_name)
        
        return data


if __name__ == "__main__":
    # Example usage
    fetcher = DataFetcher(data_dir='../data_storage')
    
    # Fetch data for all symbols and timeframes
    for symbol in ['BTCUSD', 'XAUUSD', 'EURUSD']:
        for timeframe in ['H1', 'M15']:
            try:
                data = fetcher.fetch_data(symbol, timeframe)
                print(f"Fetched {symbol} {timeframe} data: {len(data)} rows")
                print(data.head())
            except Exception as e:
                print(f"Error fetching {symbol} {timeframe}: {e}")
