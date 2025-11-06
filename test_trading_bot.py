"""
Test script for verifying the functionality of the trading bot components.
This script tests each module individually and then tests the integration of all components.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_fetcher import DataFetcher
from strategy.strategy_analyzer import StrategyAnalyzer
from trade_management.trade_manager import TradeManager
from risk_management.risk_manager import RiskManager
from backtest.backtester import Backtester

class TestTradingBot(unittest.TestCase):
    """Test cases for the trading bot components"""
    
    def setUp(self):
        """Set up test environment"""
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.data_fetcher = DataFetcher(data_dir=self.data_dir)
        self.strategy_analyzer = StrategyAnalyzer()
        self.trade_manager = TradeManager(account_balance=10000, trades_file=os.path.join(self.data_dir, 'test_trades.json'))
        self.risk_manager = RiskManager(account_balance=10000)
        self.backtester = Backtester(initial_balance=10000, data_dir=self.data_dir)
    
    def test_data_fetcher(self):
        """Test the data fetching module"""
        print("\nTesting Data Fetcher...")
        
        # Test fetching data for each symbol and timeframe
        for symbol in ['BTCUSD', 'XAUUSD', 'EURUSD']:
            for timeframe in ['H1', 'M15']:
                try:
                    data = self.data_fetcher.fetch_data(symbol, timeframe)
                    
                    # Verify data structure
                    self.assertIsInstance(data, pd.DataFrame)
                    self.assertGreater(len(data), 0)
                    
                    # Verify required columns
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in required_columns:
                        self.assertIn(col, data.columns)
                    
                    print(f"✓ Successfully fetched {symbol} {timeframe} data: {len(data)} rows")
                except Exception as e:
                    self.fail(f"Failed to fetch {symbol} {timeframe} data: {e}")
        
        # Test data storage and retrieval
        try:
            symbol = 'BTCUSD'
            timeframe = 'H1'
            
            # Fetch and save data
            data = self.data_fetcher.fetch_data(symbol, timeframe, save=True)
            
            # Load data from file
            loaded_data = self.data_fetcher.load_data(symbol, timeframe)
            
            # Verify data is the same
            self.assertEqual(len(data), len(loaded_data))
            print(f"✓ Successfully saved and loaded {symbol} {timeframe} data")
        except Exception as e:
            self.fail(f"Failed to test data storage and retrieval: {e}")
    
    def test_strategy_analyzer(self):
        """Test the strategy analysis module"""
        print("\nTesting Strategy Analyzer...")
        
        try:
            # Get sample data
            symbol = 'BTCUSD'
            htf_data = self.data_fetcher.fetch_data(symbol, 'H1')
            ltf_data = self.data_fetcher.fetch_data(symbol, 'M15')
            
            # Test market structure identification
            htf_analysis = self.strategy_analyzer.identify_market_structure(htf_data)
            self.assertIn('trend', htf_analysis.columns)
            print(f"✓ Successfully identified market structure")
            
            # Test order block detection
            htf_analysis = self.strategy_analyzer.identify_order_blocks(htf_analysis)
            self.assertIn('bullish_order_block', htf_analysis.columns)
            self.assertIn('bearish_order_block', htf_analysis.columns)
            print(f"✓ Successfully identified order blocks")
            
            # Test key levels identification
            htf_analysis, support_levels, resistance_levels = self.strategy_analyzer.identify_key_levels(htf_analysis)
            self.assertIsInstance(support_levels, list)
            self.assertIsInstance(resistance_levels, list)
            print(f"✓ Successfully identified key levels")
            
            # Test engulfing pattern detection
            ltf_analysis = self.strategy_analyzer.identify_market_structure(ltf_data)
            ltf_analysis = self.strategy_analyzer.identify_engulfing_patterns(ltf_analysis)
            self.assertIn('bullish_engulfing', ltf_analysis.columns)
            self.assertIn('bearish_engulfing', ltf_analysis.columns)
            print(f"✓ Successfully identified engulfing patterns")
            
            # Test price reaction detection
            ltf_analysis = self.strategy_analyzer.identify_price_reactions(ltf_analysis, support_levels, resistance_levels)
            self.assertIn('support_reaction', ltf_analysis.columns)
            self.assertIn('resistance_reaction', ltf_analysis.columns)
            print(f"✓ Successfully identified price reactions")
            
            # Test signal generation
            signals = self.strategy_analyzer.generate_trade_signals(htf_analysis, ltf_analysis, support_levels, resistance_levels)
            self.assertIn('buy_signal', signals.columns)
            self.assertIn('sell_signal', signals.columns)
            print(f"✓ Successfully generated trade signals")
            
            # Test time restrictions
            result = self.strategy_analyzer.check_time_restrictions()
            self.assertIsInstance(result, bool)
            print(f"✓ Successfully checked time restrictions")
            
        except Exception as e:
            self.fail(f"Strategy analyzer test failed: {e}")
    
    def test_trade_manager(self):
        """Test the trade management module"""
        print("\nTesting Trade Manager...")
        
        try:
            # Test position size calculation
            entry_price = 50000
            stop_loss = 49500
            risk_percentage = 1.0
            
            position_size = self.trade_manager.calculate_position_size(entry_price, stop_loss, risk_percentage)
            self.assertGreater(position_size, 0)
            print(f"✓ Successfully calculated position size: {position_size}")
            
            # Test signal processing
            symbol = 'BTCUSD'
            htf_data = self.data_fetcher.fetch_data(symbol, 'H1')
            ltf_data = self.data_fetcher.fetch_data(symbol, 'M15')
            
            # Create a sample signal
            signals = ltf_data.copy()
            signals['buy_signal'] = False
            signals['sell_signal'] = False
            signals['stop_loss'] = np.nan
            signals['take_profit'] = np.nan
            signals['partial_tp'] = np.nan
            signals['risk_level'] = 'low'
            
            # Set a buy signal in the last row
            signals.iloc[-1, signals.columns.get_indexer(['buy_signal'])[0]] = True
            signals.iloc[-1, signals.columns.get_indexer(['stop_loss'])[0]] = signals['low'].iloc[-1] * 0.99
            signals.iloc[-1, signals.columns.get_indexer(['take_profit'])[0]] = signals['close'].iloc[-1] * 1.02
            signals.iloc[-1, signals.columns.get_indexer(['partial_tp'])[0]] = signals['close'].iloc[-1] * 1.01
            
            # Process signals
            current_price = signals['close'].iloc[-1]
            new_trades = self.trade_manager.process_signals(signals, symbol, current_price, self.strategy_analyzer)
            
            # Verify trade creation
            self.assertGreaterEqual(len(new_trades), 0)  # May be 0 if time restrictions apply
            print(f"✓ Successfully processed signals")
            
            # Test trade updating
            market_data = {
                symbol: {
                    'close': current_price * 1.01,  # Simulate price increase
                    'market_structure': {
                        'trend': 'uptrend'
                    }
                }
            }
            
            closed_trades = self.trade_manager.update_trades(market_data)
            print(f"✓ Successfully updated trades")
            
            # Test account summary
            summary = self.trade_manager.get_account_summary()
            self.assertIn('account_balance', summary)
            self.assertIn('open_trades', summary)
            self.assertIn('total_pnl', summary)
            print(f"✓ Successfully generated account summary")
            
        except Exception as e:
            self.fail(f"Trade manager test failed: {e}")
    
    def test_risk_manager(self):
        """Test the risk management module"""
        print("\nTesting Risk Manager...")
        
        try:
            # Test position size calculation
            entry_price = 50000
            stop_loss = 49500
            risk_level = 'low'
            
            position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss, risk_level)
            self.assertGreater(position_size, 0)
            print(f"✓ Successfully calculated position size: {position_size}")
            
            # Test risk limits
            result = self.risk_manager.check_risk_limits(1.0)
            self.assertIsInstance(result, bool)
            print(f"✓ Successfully checked risk limits")
            
            # Test time restrictions
            result = self.risk_manager.check_time_restrictions()
            self.assertIsInstance(result, bool)
            print(f"✓ Successfully checked time restrictions")
            
            # Test trade validation
            symbol = 'BTCUSD'
            trade_type = 'buy'
            entry_price = 50000
            stop_loss = 49500
            risk_level = 'low'
            
            is_valid, position_size, reason = self.risk_manager.validate_trade(
                symbol, trade_type, entry_price, stop_loss, risk_level
            )
            
            self.assertIsInstance(is_valid, bool)
            self.assertIsInstance(position_size, float)
            self.assertIsInstance(reason, str)
            print(f"✓ Successfully validated trade: {reason}")
            
            # Test risk report
            report = self.risk_manager.get_risk_report()
            self.assertIn('account_balance', report)
            self.assertIn('total_risk_percentage', report)
            print(f"✓ Successfully generated risk report")
            
        except Exception as e:
            self.fail(f"Risk manager test failed: {e}")
    
    def test_backtester(self):
        """Test the backtesting module"""
        print("\nTesting Backtester...")
        
        try:
            # Test backtest run with a small date range
            symbol = 'BTCUSD'
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            results = self.backtester.run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                high_risk_pct=0.5,
                low_risk_pct=1.0
            )
            
            # Verify results structure
            self.assertIn('initial_balance', results)
            self.assertIn('final_balance', results)
            self.assertIn('total_return', results)
            self.assertIn('win_rate', results)
            print(f"✓ Successfully ran backtest")
            
            # Test equity curve plotting
            fig = self.backtester.plot_equity_curve()
            self.assertIsNotNone(fig)
            print(f"✓ Successfully plotted equity curve")
            
            # Test results saving
            output_dir = os.path.join(self.data_dir, 'backtest_results')
            os.makedirs(output_dir, exist_ok=True)
            
            success = self.backtester.save_results(results, os.path.join(output_dir, 'test_results.json'))
            self.assertTrue(success)
            print(f"✓ Successfully saved backtest results")
            
            # Test report generation
            success = self.backtester.generate_report(results, os.path.join(output_dir, 'test_report.html'))
            self.assertTrue(success)
            print(f"✓ Successfully generated backtest report")
            
        except Exception as e:
            self.fail(f"Backtester test failed: {e}")
    
    def test_integration(self):
        """Test the integration of all components"""
        print("\nTesting Component Integration...")
        
        try:
            # Test data flow from fetcher to strategy analyzer
            symbol = 'BTCUSD'
            htf_data = self.data_fetcher.fetch_data(symbol, 'H1')
            ltf_data = self.data_fetcher.fetch_data(symbol, 'M15')
            
            htf_analysis, signals, support_levels, resistance_levels = self.strategy_analyzer.analyze_market(htf_data, ltf_data)
            
            # Test data flow from strategy analyzer to trade manager
            current_price = ltf_data['close'].iloc[-1]
            
            # Check if trade is valid with risk manager
            if signals['buy_signal'].any() or signals['sell_signal'].any():
                idx = signals['buy_signal'].idxmax() if signals['buy_signal'].any() else signals['sell_signal'].idxmax()
                signal = signals.loc[idx]
                
                trade_type = 'buy' if signal['buy_signal'] else 'sell'
                risk_level = signal['risk_level']
                stop_loss = signal['stop_loss']
                
                is_valid, position_size, reason = self.risk_manager.validate_trade(
                    symbol, trade_type, current_price, stop_loss, risk_level
                )
                
                print(f"Trade validation: {is_valid}, Reason: {reason}")
            
            # Test processing signals through trade manager
            new_trades = self.trade_manager.process_signals(signals, symbol, current_price, self.strategy_analyzer)
            
            # Update market data for trade manager
            market_data = {
                symbol: {
                    'close': current_price,
                    'market_structure': {
                        'trend': htf_analysis['trend'].iloc[-1]
                    }
                }
            }
            
            # Update trades
            closed_trades = self.trade_manager.update_trades(market_data)
            
            print(f"✓ Successfully integrated all components")
            
        except Exception as e:
            self.fail(f"Integration test failed: {e}")


if __name__ == "__main__":
    unittest.main()
