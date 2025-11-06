"""
Backtesting Module for Trading Bot

This module provides functionality for backtesting the trading strategy
on historical data to evaluate its performance before live trading.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
from datetime import datetime, timedelta
import json

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_fetcher import DataFetcher
from strategy.strategy_analyzer import StrategyAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtest')

class Backtester:
    """
    Class for backtesting the trading strategy on historical data.
    """
    
    def __init__(self, initial_balance=10000, data_dir=None):
        """
        Initialize the Backtester.
        
        Args:
            initial_balance (float): Initial account balance for backtesting
            data_dir (str): Directory to store/load data
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_storage')
        else:
            self.data_dir = data_dir
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.data_fetcher = DataFetcher(data_dir=self.data_dir)
        self.strategy_analyzer = StrategyAnalyzer()
        
        self.trades = []
        self.equity_curve = []
        
        logger.info(f"Backtester initialized with balance: ${initial_balance}")
    
    def load_historical_data(self, symbol, start_date, end_date=None):
        """
        Load historical data for backtesting.
        
        Args:
            symbol (str): Market symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format (defaults to today)
            
        Returns:
            tuple: (HTF data, LTF data)
        """
        logger.info(f"Loading historical data for {symbol} from {start_date} to {end_date or 'today'}")
        
        # Convert dates to datetime objects
        start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = pd.to_datetime('today')
        
        # Calculate period based on date range
        days_diff = (end_date - start_date).days
        if days_diff <= 30:
            period = '1mo'
        elif days_diff <= 90:
            period = '3mo'
        elif days_diff <= 180:
            period = '6mo'
        elif days_diff <= 365:
            period = '1y'
        else:
            period = '2y'
        
        # Fetch data
        symbol_info = self.data_fetcher.symbols.get(symbol, {})
        if not symbol_info:
            raise ValueError(f"Unsupported symbol: {symbol}")
        
        if symbol_info['source'] == 'yfinance':
            # For yfinance data
            import yfinance as yf
            
            ticker = yf.Ticker(symbol_info['symbol'])
            
            # Fetch H1 data
            htf_data = ticker.history(start=start_date, end=end_date, interval='1h')
            htf_data.columns = [col.lower() for col in htf_data.columns]
            
            # Fetch M15 data
            ltf_data = ticker.history(start=start_date, end=end_date, interval='15m')
            ltf_data.columns = [col.lower() for col in ltf_data.columns]
            
        elif symbol_info['source'] == 'ccxt':
            # For cryptocurrency data
            import ccxt
            
            exchange = ccxt.binance({
                'enableRateLimit': True,
            })
            
            # For backtesting with CCXT, we need to fetch data in chunks
            # This is a simplified version - in a real implementation, you would
            # need to handle pagination and rate limits more carefully
            
            # Fetch H1 data
            htf_data = self._fetch_ccxt_historical_data(
                exchange, symbol_info['symbol'], '1h', start_date, end_date
            )
            
            # Fetch M15 data
            ltf_data = self._fetch_ccxt_historical_data(
                exchange, symbol_info['symbol'], '15m', start_date, end_date
            )
        
        else:
            raise ValueError(f"Unsupported data source: {symbol_info['source']}")
        
        logger.info(f"Loaded {len(htf_data)} H1 candles and {len(ltf_data)} M15 candles")
        
        return htf_data, ltf_data
    
    def _fetch_ccxt_historical_data(self, exchange, symbol, timeframe, start_date, end_date):
        """
        Fetch historical data from CCXT exchange.
        
        Args:
            exchange: CCXT exchange instance
            symbol (str): Market symbol
            timeframe (str): Timeframe ('1h' or '15m')
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        # Convert dates to milliseconds
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)
        
        all_candles = []
        current_since = since
        
        # Fetch data in chunks
        while current_since < until:
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update since for next iteration
                current_since = candles[-1][0] + 1
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def run_backtest(self, symbol, start_date, end_date=None, high_risk_pct=0.5, low_risk_pct=1.0, max_open_trades=2):
        """
        Run backtest on historical data.
        
        Args:
            symbol (str): Market symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format (defaults to today)
            high_risk_pct (float): Risk percentage for high-risk trades
            low_risk_pct (float): Risk percentage for low-risk trades
            max_open_trades (int): Maximum number of open trades
            
        Returns:
            dict: Backtest results
        """
        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date or 'today'}")
        
        # Load historical data
        htf_data, ltf_data = self.load_historical_data(symbol, start_date, end_date)
        
        # Reset backtest state
        self.current_balance = self.initial_balance
        self.trades = []
        self.equity_curve = [{'date': ltf_data.index[0], 'balance': self.current_balance}]
        open_trades = []
        
        # Analyze HTF data first (this doesn't change as frequently)
        htf_analysis = self.strategy_analyzer.identify_market_structure(htf_data)
        htf_analysis = self.strategy_analyzer.identify_order_blocks(htf_analysis)
        htf_analysis, support_levels, resistance_levels = self.strategy_analyzer.identify_key_levels(htf_analysis)
        
        # Process each LTF candle
        for i in range(100, len(ltf_data)):  # Start after 100 candles to have enough data for indicators
            current_time = ltf_data.index[i]
            
            # Skip weekends for forex
            if symbol in ['EURUSD'] and current_time.weekday() >= 5:
                continue
            
            # Check time restrictions (Friday US session)
            if current_time.weekday() == 4 and 13 <= current_time.hour < 20:
                continue
            
            # Get current HTF candle
            htf_time = current_time.floor('H')
            htf_idx = htf_analysis.index.get_indexer([htf_time], method='nearest')[0]
            current_htf = htf_analysis.iloc[htf_idx]
            
            # Get LTF data up to current candle
            current_ltf_data = ltf_data.iloc[:i+1]
            
            # Analyze LTF data
            ltf_analysis = self.strategy_analyzer.identify_market_structure(current_ltf_data)
            ltf_analysis = self.strategy_analyzer.identify_order_blocks(ltf_analysis)
            ltf_analysis = self.strategy_analyzer.identify_engulfing_patterns(ltf_analysis)
            ltf_analysis = self.strategy_analyzer.identify_price_reactions(ltf_analysis, support_levels, resistance_levels)
            
            # Get current candle
            current_candle = ltf_analysis.iloc[-1]
            current_price = current_candle['close']
            
            # Update open trades
            closed_trades = []
            for trade in open_trades:
                # Update trade P&L
                if trade['type'] == 'buy':
                    trade['current_price'] = current_price
                    trade['pnl'] = (current_price - trade['entry_price']) * trade['position_size']
                    trade['pnl_percentage'] = (current_price - trade['entry_price']) / trade['entry_price'] * 100
                else:  # sell
                    trade['current_price'] = current_price
                    trade['pnl'] = (trade['entry_price'] - current_price) * trade['position_size']
                    trade['pnl_percentage'] = (trade['entry_price'] - current_price) / trade['entry_price'] * 100
                
                # Check for partial take profit
                if not trade.get('partial_exit_executed', False):
                    if (trade['type'] == 'buy' and current_price >= trade['partial_tp']) or \
                       (trade['type'] == 'sell' and current_price <= trade['partial_tp']):
                        # Execute partial exit (50% of position)
                        partial_size = trade['position_size'] * 0.5
                        trade['position_size'] -= partial_size
                        
                        # Calculate partial profit
                        if trade['type'] == 'buy':
                            partial_profit = (current_price - trade['entry_price']) * partial_size
                        else:  # sell
                            partial_profit = (trade['entry_price'] - current_price) * partial_size
                        
                        # Update balance
                        self.current_balance += partial_profit
                        
                        # Mark partial exit as executed
                        trade['partial_exit_executed'] = True
                        trade['partial_exit_price'] = current_price
                        trade['partial_exit_time'] = current_time
                        
                        logger.info(f"Executed partial take profit at {current_time}, profit: ${partial_profit:.2f}")
                
                # Check for stop loss
                if (trade['type'] == 'buy' and current_price <= trade['stop_loss']) or \
                   (trade['type'] == 'sell' and current_price >= trade['stop_loss']):
                    # Close trade at stop loss
                    if trade['type'] == 'buy':
                        realized_pnl = (trade['stop_loss'] - trade['entry_price']) * trade['position_size']
                    else:  # sell
                        realized_pnl = (trade['entry_price'] - trade['stop_loss']) * trade['position_size']
                    
                    # Update balance
                    self.current_balance += realized_pnl
                    
                    # Add to closed trades
                    trade['exit_time'] = current_time
                    trade['exit_price'] = trade['stop_loss']
                    trade['exit_reason'] = 'stop_loss'
                    trade['realized_pnl'] = realized_pnl
                    self.trades.append(trade)
                    
                    # Mark for removal from open trades
                    closed_trades.append(trade)
                    
                    logger.info(f"Closed trade at stop loss, P&L: ${realized_pnl:.2f}")
                
                # Check for take profit
                elif (trade['type'] == 'buy' and current_price >= trade['take_profit']) or \
                     (trade['type'] == 'sell' and current_price <= trade['take_profit']):
                    # Close trade at take profit
                    if trade['type'] == 'buy':
                        realized_pnl = (trade['take_profit'] - trade['entry_price']) * trade['position_size']
                    else:  # sell
                        realized_pnl = (trade['entry_price'] - trade['take_profit']) * trade['position_size']
                    
                    # Update balance
                    self.current_balance += realized_pnl
                    
                    # Add to closed trades
                    trade['exit_time'] = current_time
                    trade['exit_price'] = trade['take_profit']
                    trade['exit_reason'] = 'take_profit'
                    trade['realized_pnl'] = realized_pnl
                    self.trades.append(trade)
                    
                    # Mark for removal from open trades
                    closed_trades.append(trade)
                    
                    logger.info(f"Closed trade at take profit, P&L: ${realized_pnl:.2f}")
                
                # Check for structure invalidation (opposite trend)
                elif (trade['type'] == 'buy' and current_htf['trend'] == 'downtrend') or \
                     (trade['type'] == 'sell' and current_htf['trend'] == 'uptrend'):
                    # Close trade due to structure invalidation
                    realized_pnl = trade['pnl']
                    
                    # Update balance
                    self.current_balance += realized_pnl
                    
                    # Add to closed trades
                    trade['exit_time'] = current_time
                    trade['exit_price'] = current_price
                    trade['exit_reason'] = 'structure_invalidation'
                    trade['realized_pnl'] = realized_pnl
                    self.trades.append(trade)
                    
                    # Mark for removal from open trades
                    closed_trades.append(trade)
                    
                    logger.info(f"Closed trade due to structure invalidation, P&L: ${realized_pnl:.2f}")
            
            # Remove closed trades from open trades
            for trade in closed_trades:
                open_trades.remove(trade)
            
            # Check for new trade signals
            if len(open_trades) < max_open_trades:
                # Buy signal
                if current_candle['close'] > current_candle['open'] and \
                   ((current_htf['trend'] == 'uptrend' and 
                     (ltf_analysis['bullish_engulfing'].iloc[-1] or ltf_analysis['support_reaction'].iloc[-1]))):
                    
                    # Determine risk level
                    risk_level = 'low' if (ltf_analysis['support_reaction'].iloc[-1] or 
                                          ltf_analysis['bullish_order_block'].iloc[-5:].any()) else 'high'
                    risk_percentage = low_risk_pct if risk_level == 'low' else high_risk_pct
                    
                    # F
(Content truncated due to size limit. Use line ranges to read in chunks)