"""
Trade Management Module for Trading Bot

This module is responsible for:
- Managing trade entries based on signals
- Setting stop loss and take profit levels
- Handling partial take profits
- Managing trade exits
- Tracking open positions
- Enforcing maximum open trades limit
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('trade_manager')

class TradeManager:
    """
    Class for managing trades based on signals from the strategy analyzer.
    """
    
    def __init__(self, account_balance=10000, max_open_trades=2, trades_file='../data_storage/trades.json'):
        """
        Initialize the TradeManager.
        
        Args:
            account_balance (float): Initial account balance
            max_open_trades (int): Maximum number of open trades allowed
            trades_file (str): File to store trade data
        """
        self.account_balance = account_balance
        self.max_open_trades = max_open_trades
        self.trades_file = trades_file
        self.open_trades = []
        self.closed_trades = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(trades_file), exist_ok=True)
        
        # Load existing trades if file exists
        if os.path.exists(trades_file):
            self.load_trades()
        
        logger.info(f"TradeManager initialized with balance: ${account_balance}")
    
    def load_trades(self):
        """Load trades from file."""
        try:
            with open(self.trades_file, 'r') as f:
                data = json.load(f)
                self.account_balance = data.get('account_balance', self.account_balance)
                self.open_trades = data.get('open_trades', [])
                self.closed_trades = data.get('closed_trades', [])
            logger.info(f"Loaded {len(self.open_trades)} open trades and {len(self.closed_trades)} closed trades")
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
    
    def save_trades(self):
        """Save trades to file."""
        try:
            data = {
                'account_balance': self.account_balance,
                'open_trades': self.open_trades,
                'closed_trades': self.closed_trades
            }
            with open(self.trades_file, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info("Trades saved to file")
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def calculate_position_size(self, entry_price, stop_loss, risk_percentage):
        """
        Calculate position size based on risk percentage and stop loss.
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            risk_percentage (float): Risk percentage (0.5% or 1%)
            
        Returns:
            float: Position size in units
        """
        # Calculate risk amount in dollars
        risk_amount = self.account_balance * (risk_percentage / 100)
        
        # Calculate risk per unit
        if entry_price > stop_loss:  # Long position
            risk_per_unit = entry_price - stop_loss
        else:  # Short position
            risk_per_unit = stop_loss - entry_price
        
        # Calculate position size
        position_size = risk_amount / risk_per_unit
        
        logger.info(f"Calculated position size: {position_size} units (risk: {risk_percentage}%)")
        return position_size
    
    def check_time_restrictions(self, analyzer, current_time=None):
        """
        Check if current time is within trading restrictions.
        
        Args:
            analyzer: Strategy analyzer instance
            current_time (datetime): Current time
            
        Returns:
            bool: True if trading is allowed, False if restricted
        """
        return analyzer.check_time_restrictions(current_time)
    
    def process_signals(self, signals, symbol, current_price, analyzer, current_time=None):
        """
        Process trade signals and execute trades if conditions are met.
        
        Args:
            signals (pd.DataFrame): DataFrame with trade signals
            symbol (str): Market symbol
            current_price (float): Current market price
            analyzer: Strategy analyzer instance
            current_time (datetime): Current time
            
        Returns:
            list: New trades that were opened
        """
        # Check time restrictions
        if not self.check_time_restrictions(analyzer, current_time):
            logger.info("Trading restricted due to time restrictions")
            return []
        
        # Check if we can open more trades
        if len(self.open_trades) >= self.max_open_trades:
            logger.info(f"Maximum number of open trades ({self.max_open_trades}) reached")
            return []
        
        # Get the latest signals
        latest_signals = signals.iloc[-1]
        
        new_trades = []
        
        # Process buy signal
        if latest_signals['buy_signal']:
            # Determine risk percentage based on risk level
            risk_percentage = 1.0 if latest_signals['risk_level'] == 'low' else 0.5
            
            # Calculate position size
            position_size = self.calculate_position_size(
                current_price, 
                latest_signals['stop_loss'], 
                risk_percentage
            )
            
            # Create trade
            trade = {
                'id': len(self.open_trades) + len(self.closed_trades) + 1,
                'symbol': symbol,
                'type': 'buy',
                'entry_time': datetime.now().isoformat(),
                'entry_price': current_price,
                'position_size': position_size,
                'stop_loss': latest_signals['stop_loss'],
                'take_profit': latest_signals['take_profit'],
                'partial_tp': latest_signals['partial_tp'],
                'risk_percentage': risk_percentage,
                'partial_exit_executed': False,
                'current_price': current_price,
                'pnl': 0.0,
                'pnl_percentage': 0.0
            }
            
            # Add to open trades
            self.open_trades.append(trade)
            new_trades.append(trade)
            
            logger.info(f"Opened BUY trade for {symbol} at {current_price}")
        
        # Process sell signal
        elif latest_signals['sell_signal']:
            # Determine risk percentage based on risk level
            risk_percentage = 1.0 if latest_signals['risk_level'] == 'low' else 0.5
            
            # Calculate position size
            position_size = self.calculate_position_size(
                current_price, 
                latest_signals['stop_loss'], 
                risk_percentage
            )
            
            # Create trade
            trade = {
                'id': len(self.open_trades) + len(self.closed_trades) + 1,
                'symbol': symbol,
                'type': 'sell',
                'entry_time': datetime.now().isoformat(),
                'entry_price': current_price,
                'position_size': position_size,
                'stop_loss': latest_signals['stop_loss'],
                'take_profit': latest_signals['take_profit'],
                'partial_tp': latest_signals['partial_tp'],
                'risk_percentage': risk_percentage,
                'partial_exit_executed': False,
                'current_price': current_price,
                'pnl': 0.0,
                'pnl_percentage': 0.0
            }
            
            # Add to open trades
            self.open_trades.append(trade)
            new_trades.append(trade)
            
            logger.info(f"Opened SELL trade for {symbol} at {current_price}")
        
        # Save trades to file
        self.save_trades()
        
        return new_trades
    
    def update_trades(self, market_data):
        """
        Update open trades with current market data.
        
        Args:
            market_data (dict): Dictionary with current market data for each symbol
            
        Returns:
            list: Trades that were closed
        """
        closed_trades = []
        
        for trade in self.open_trades[:]:  # Create a copy to iterate over
            symbol = trade['symbol']
            
            if symbol not in market_data:
                logger.warning(f"No market data for {symbol}, skipping trade update")
                continue
            
            # Get current price
            current_price = market_data[symbol]['close']
            trade['current_price'] = current_price
            
            # Calculate P&L
            if trade['type'] == 'buy':
                trade['pnl'] = (current_price - trade['entry_price']) * trade['position_size']
                trade['pnl_percentage'] = (current_price - trade['entry_price']) / trade['entry_price'] * 100
            else:  # sell
                trade['pnl'] = (trade['entry_price'] - current_price) * trade['position_size']
                trade['pnl_percentage'] = (trade['entry_price'] - current_price) / trade['entry_price'] * 100
            
            # Check for partial take profit
            if not trade['partial_exit_executed']:
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
                    
                    # Update account balance
                    self.account_balance += partial_profit
                    
                    # Mark partial exit as executed
                    trade['partial_exit_executed'] = True
                    
                    logger.info(f"Executed partial take profit for {symbol} trade, profit: ${partial_profit:.2f}")
            
            # Check for stop loss
            if (trade['type'] == 'buy' and current_price <= trade['stop_loss']) or \
               (trade['type'] == 'sell' and current_price >= trade['stop_loss']):
                # Close trade at stop loss
                if trade['type'] == 'buy':
                    realized_pnl = (trade['stop_loss'] - trade['entry_price']) * trade['position_size']
                else:  # sell
                    realized_pnl = (trade['entry_price'] - trade['stop_loss']) * trade['position_size']
                
                # Update account balance
                self.account_balance += realized_pnl
                
                # Add to closed trades
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = trade['stop_loss']
                trade['exit_reason'] = 'stop_loss'
                trade['realized_pnl'] = realized_pnl
                self.closed_trades.append(trade)
                
                # Remove from open trades
                self.open_trades.remove(trade)
                closed_trades.append(trade)
                
                logger.info(f"Closed {symbol} trade at stop loss, P&L: ${realized_pnl:.2f}")
            
            # Check for take profit
            elif (trade['type'] == 'buy' and current_price >= trade['take_profit']) or \
                 (trade['type'] == 'sell' and current_price <= trade['take_profit']):
                # Close trade at take profit
                if trade['type'] == 'buy':
                    realized_pnl = (trade['take_profit'] - trade['entry_price']) * trade['position_size']
                else:  # sell
                    realized_pnl = (trade['entry_price'] - trade['take_profit']) * trade['position_size']
                
                # Update account balance
                self.account_balance += realized_pnl
                
                # Add to closed trades
                trade['exit_time'] = datetime.now().isoformat()
                trade['exit_price'] = trade['take_profit']
                trade['exit_reason'] = 'take_profit'
                trade['realized_pnl'] = realized_pnl
                self.closed_trades.append(trade)
                
                # Remove from open trades
                self.open_trades.remove(trade)
                closed_trades.append(trade)
                
                logger.info(f"Closed {symbol} trade at take profit, P&L: ${realized_pnl:.2f}")
            
            # Check for structure invalidation (opposite trend)
            elif 'market_structure' in market_data[symbol] and 'trend' in market_data[symbol]['market_structure']:
                current_trend = market_data[symbol]['market_structure']['trend']
                
                if (trade['type'] == 'buy' and current_trend == 'downtrend') or \
                   (trade['type'] == 'sell' and current_trend == 'uptrend'):
                    # Close trade due to structure invalidation
                    realized_pnl = trade['pnl']
                    
                    # Update account balance
                    self.account_balance += realized_pnl
                    
                    # Add to closed trades
                    trade['exit_time'] = datetime.now().isoformat()
                    trade['exit_price'] = current_price
                    trade['exit_reason'] = 'structure_invalidation'
                    trade['realized_pnl'] = realized_pnl
                    self.closed_trades.append(trade)
                    
                    # Remove from open trades
                    self.open_trades.remove(trade)
                    closed_trades.append(trade)
                    
                    logger.info(f"Closed {symbol} trade due to structure invalidation, P&L: ${realized_pnl:.2f}")
        
        # Save trades to file
        self.save_trades()
        
        return closed_trades
    
    def get_account_summary(self):
        """
        Get account summary.
        
        Returns:
            dict: Account summary
        """
        # Calculate total P&L
        total_open_pnl = sum(trade['pnl'] for trade in self.open_trades)
        total_closed_pnl = sum(trade.get('realized_pnl', 0) for trade in self.closed_trades)
        total_pnl = total_open_pnl + total_closed_pnl
        
        # Calculate win rate
        winning_trades = [t for t in self.closed_trades if t.get('realized_pnl', 0) > 0]
        win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0
        
        return {
            'account_balance': self.account_balance,
            'open_trades': len(self.open_trades),
            'closed_trades': len(self.closed_trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'max_open_trades': self.max_open_trades
        }


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data.data_fetcher import DataFetcher
    from strategy.strategy_analyzer import StrategyAnalyzer
    
    # Initialize components
    fetcher = DataFetcher(data_dir='../data_storage')
    analyzer = StrategyAnalyzer()
    manager = TradeManager(account_balance=10000)
    
    # Fetch data
    symbol = '
(Content truncated due to size limit. Use line ranges to read in chunks)