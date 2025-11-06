"""
Risk Management Module for Trading Bot

This module is responsible for:
- Managing risk across all trades
- Enforcing position sizing based on risk percentage
- Monitoring account risk exposure
- Implementing time-based trading restrictions
- Preventing trades during high-risk periods
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import os
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('risk_manager')

class RiskManager:
    """
    Class for managing risk across all trades in the trading bot.
    """
    
    def __init__(self, account_balance=10000, max_open_trades=2, max_total_risk=2.0):
        """
        Initialize the RiskManager.
        
        Args:
            account_balance (float): Initial account balance
            max_open_trades (int): Maximum number of open trades allowed
            max_total_risk (float): Maximum total risk as percentage of account
        """
        self.account_balance = account_balance
        self.max_open_trades = max_open_trades
        self.max_total_risk = max_total_risk
        self.open_trades = []
        self.risk_levels = {
            'high': 0.5,  # 0.5% risk for high-risk trades
            'low': 1.0    # 1.0% risk for low-risk trades
        }
        self.economic_calendar = []
        
        logger.info(f"RiskManager initialized with balance: ${account_balance}, max risk: {max_total_risk}%")
    
    def update_account_balance(self, new_balance):
        """
        Update the account balance.
        
        Args:
            new_balance (float): New account balance
        """
        self.account_balance = new_balance
        logger.info(f"Account balance updated to ${new_balance}")
    
    def update_open_trades(self, open_trades):
        """
        Update the list of open trades.
        
        Args:
            open_trades (list): List of open trades
        """
        self.open_trades = open_trades
        logger.info(f"Open trades updated, current count: {len(open_trades)}")
    
    def calculate_position_size(self, entry_price, stop_loss, risk_level):
        """
        Calculate position size based on risk level and stop loss.
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            risk_level (str): Risk level ('high' or 'low')
            
        Returns:
            float: Position size in units
        """
        # Get risk percentage based on risk level
        risk_percentage = self.risk_levels.get(risk_level, 0.5)
        
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
    
    def calculate_total_risk(self):
        """
        Calculate total risk exposure from all open trades.
        
        Returns:
            float: Total risk as percentage of account
        """
        total_risk = sum(trade.get('risk_percentage', 0) for trade in self.open_trades)
        logger.info(f"Total risk exposure: {total_risk}%")
        return total_risk
    
    def check_risk_limits(self, new_trade_risk):
        """
        Check if adding a new trade would exceed risk limits.
        
        Args:
            new_trade_risk (float): Risk percentage of the new trade
            
        Returns:
            bool: True if within limits, False if exceeds limits
        """
        # Check number of open trades
        if len(self.open_trades) >= self.max_open_trades:
            logger.warning(f"Maximum number of open trades ({self.max_open_trades}) reached")
            return False
        
        # Check total risk
        current_risk = self.calculate_total_risk()
        total_risk = current_risk + new_trade_risk
        
        if total_risk > self.max_total_risk:
            logger.warning(f"Adding trade would exceed maximum risk: {total_risk}% > {self.max_total_risk}%")
            return False
        
        logger.info(f"Trade within risk limits: {total_risk}% <= {self.max_total_risk}%")
        return True
    
    def check_time_restrictions(self, current_time=None):
        """
        Check if current time is within trading restrictions.
        
        Args:
            current_time (datetime): Current time (uses system time if None)
            
        Returns:
            bool: True if trading is allowed, False if restricted
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Check if it's Friday during US session (13:00 to 20:00 UTC)
        if current_time.weekday() == 4:  # Friday
            if 13 <= current_time.hour < 20:
                logger.info("Trading restricted: Friday during US session")
                return False
        
        # Check for upcoming news events (within 30 minutes)
        for event in self.economic_calendar:
            event_time = datetime.fromisoformat(event['time'])
            time_diff = (event_time - current_time).total_seconds() / 60
            
            if 0 <= time_diff <= 30:  # Event is within the next 30 minutes
                if event['impact'] == 'high':
                    logger.info(f"Trading restricted: High-impact news event in {time_diff:.1f} minutes")
                    return False
        
        return True
    
    def update_economic_calendar(self, days=7):
        """
        Update the economic calendar with upcoming events.
        
        Args:
            days (int): Number of days to fetch
            
        Note: In a real implementation, this would connect to an economic calendar API.
        For this example, we'll simulate some events.
        """
        # In a real implementation, this would fetch data from an API
        # For this example, we'll create some simulated events
        
        self.economic_calendar = []
        current_time = datetime.utcnow()
        
        # Simulate some high-impact events
        events = [
            {
                'time': (current_time + timedelta(hours=2)).isoformat(),
                'title': 'US Non-Farm Payrolls',
                'impact': 'high',
                'currency': 'USD'
            },
            {
                'time': (current_time + timedelta(days=1, hours=14)).isoformat(),
                'title': 'FOMC Statement',
                'impact': 'high',
                'currency': 'USD'
            },
            {
                'time': (current_time + timedelta(days=2, hours=8)).isoformat(),
                'title': 'ECB Interest Rate Decision',
                'impact': 'high',
                'currency': 'EUR'
            }
        ]
        
        self.economic_calendar = events
        logger.info(f"Economic calendar updated with {len(events)} events")
        
        return events
    
    def validate_trade(self, symbol, trade_type, entry_price, stop_loss, risk_level, current_time=None):
        """
        Validate if a trade meets all risk management criteria.
        
        Args:
            symbol (str): Market symbol
            trade_type (str): Trade type ('buy' or 'sell')
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            risk_level (str): Risk level ('high' or 'low')
            current_time (datetime): Current time
            
        Returns:
            tuple: (is_valid, position_size, reason)
        """
        # Check time restrictions
        if not self.check_time_restrictions(current_time):
            return False, 0, "Time restrictions in effect"
        
        # Get risk percentage based on risk level
        risk_percentage = self.risk_levels.get(risk_level, 0.5)
        
        # Check risk limits
        if not self.check_risk_limits(risk_percentage):
            return False, 0, "Exceeds risk limits"
        
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss, risk_level)
        
        # Validate stop loss distance (prevent extremely tight stops)
        stop_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
        if stop_distance_pct < 0.1:  # Minimum 0.1% stop distance
            return False, 0, "Stop loss too close to entry"
        
        # Check for existing trades on the same symbol
        for trade in self.open_trades:
            if trade['symbol'] == symbol:
                return False, 0, f"Already have an open trade for {symbol}"
        
        return True, position_size, "Trade validated"
    
    def get_risk_report(self):
        """
        Generate a risk report.
        
        Returns:
            dict: Risk report
        """
        total_risk = self.calculate_total_risk()
        
        report = {
            'account_balance': self.account_balance,
            'open_trades': len(self.open_trades),
            'total_risk_percentage': total_risk,
            'max_risk_percentage': self.max_total_risk,
            'risk_utilization': (total_risk / self.max_total_risk) * 100 if self.max_total_risk > 0 else 0,
            'trades_by_symbol': {},
            'trades_by_risk_level': {
                'high': 0,
                'low': 0
            }
        }
        
        # Count trades by symbol and risk level
        for trade in self.open_trades:
            symbol = trade['symbol']
            risk_level = 'low' if trade.get('risk_percentage', 0) >= 1.0 else 'high'
            
            if symbol not in report['trades_by_symbol']:
                report['trades_by_symbol'][symbol] = 0
            
            report['trades_by_symbol'][symbol] += 1
            report['trades_by_risk_level'][risk_level] += 1
        
        return report


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from trade_management.trade_manager import TradeManager
    
    # Initialize components
    risk_manager = RiskManager(account_balance=10000)
    trade_manager = TradeManager(account_balance=10000)
    
    # Update economic calendar
    risk_manager.update_economic_calendar()
    
    # Validate a potential trade
    symbol = 'BTCUSD'
    trade_type = 'buy'
    entry_price = 50000
    stop_loss = 49500
    risk_level = 'low'
    
    is_valid, position_size, reason = risk_manager.validate_trade(
        symbol, trade_type, entry_price, stop_loss, risk_level
    )
    
    print(f"Trade validation: {is_valid}")
    print(f"Position size: {position_size}")
    print(f"Reason: {reason}")
    
    # Generate risk report
    report = risk_manager.get_risk_report()
    print("\nRisk Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
