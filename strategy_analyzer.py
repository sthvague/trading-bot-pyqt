"""
Strategy Analyzer Module for Trading Bot

This module is responsible for analyzing market data to identify:
- Market structure (trends, ranges)
- Order blocks (zones where sharp moves started)
- Key levels (support/resistance)
- Engulfing patterns and price reactions

It implements the price action strategy specified in the requirements.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('strategy_analyzer')

class StrategyAnalyzer:
    """
    Class for analyzing market data and identifying trading opportunities
    based on price action strategy.
    """
    
    def __init__(self):
        """Initialize the StrategyAnalyzer"""
        logger.info("StrategyAnalyzer initialized")
    
    def identify_market_structure(self, df, window=10):
        """
        Identify market structure (uptrend, downtrend, or range).
        
        Args:
            df (pd.DataFrame): OHLCV data
            window (int): Window size for trend determination
            
        Returns:
            pd.DataFrame: DataFrame with market structure column added
        """
        logger.info("Identifying market structure")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate short-term and long-term moving averages
        result['sma_short'] = ta.sma(result['close'], length=window)
        result['sma_long'] = ta.sma(result['close'], length=window*2)
        
        # Determine trend based on moving averages
        result['trend'] = np.where(
            result['sma_short'] > result['sma_long'], 
            'uptrend',
            np.where(
                result['sma_short'] < result['sma_long'], 
                'downtrend', 
                'range'
            )
        )
        
        # Identify swing highs and lows
        result['swing_high'] = False
        result['swing_low'] = False
        
        for i in range(window, len(result) - window):
            # Check if this is a swing high
            if all(result['high'].iloc[i] > result['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(result['high'].iloc[i] > result['high'].iloc[i+j] for j in range(1, window+1)):
                result['swing_high'].iloc[i] = True
            
            # Check if this is a swing low
            if all(result['low'].iloc[i] < result['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(result['low'].iloc[i] < result['low'].iloc[i+j] for j in range(1, window+1)):
                result['swing_low'].iloc[i] = True
        
        return result
    
    def identify_order_blocks(self, df, threshold=0.5):
        """
        Identify order blocks (zones where sharp moves started).
        
        Args:
            df (pd.DataFrame): OHLCV data with market structure
            threshold (float): Minimum percentage move to consider a sharp move
            
        Returns:
            pd.DataFrame: DataFrame with order blocks identified
        """
        logger.info("Identifying order blocks")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate percentage change
        result['pct_change'] = result['close'].pct_change()
        
        # Identify sharp moves
        result['sharp_move'] = abs(result['pct_change']) > threshold/100
        
        # Initialize order block columns
        result['bullish_order_block'] = False
        result['bearish_order_block'] = False
        
        # Look for candles before sharp moves
        for i in range(1, len(result) - 1):
            # If there's a sharp upward move, the previous candle might be a bullish order block
            if result['sharp_move'].iloc[i] and result['pct_change'].iloc[i] > 0:
                result['bullish_order_block'].iloc[i-1] = True
            
            # If there's a sharp downward move, the previous candle might be a bearish order block
            if result['sharp_move'].iloc[i] and result['pct_change'].iloc[i] < 0:
                result['bearish_order_block'].iloc[i-1] = True
        
        # Calculate order block levels
        result['bullish_ob_low'] = np.where(result['bullish_order_block'], result['low'], np.nan)
        result['bullish_ob_high'] = np.where(result['bullish_order_block'], result['high'], np.nan)
        
        result['bearish_ob_low'] = np.where(result['bearish_order_block'], result['low'], np.nan)
        result['bearish_ob_high'] = np.where(result['bearish_order_block'], result['high'], np.nan)
        
        return result
    
    def identify_key_levels(self, df, window=20, tolerance=0.001):
        """
        Identify key support and resistance levels.
        
        Args:
            df (pd.DataFrame): OHLCV data
            window (int): Window size for level identification
            tolerance (float): Percentage tolerance for level proximity
            
        Returns:
            pd.DataFrame: DataFrame with key levels identified
        """
        logger.info("Identifying key levels")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Use swing highs and lows as potential key levels
        highs = result[result['swing_high']]['high'].tolist()
        lows = result[result['swing_low']]['low'].tolist()
        
        # Cluster similar levels
        support_levels = self._cluster_levels(lows, tolerance)
        resistance_levels = self._cluster_levels(highs, tolerance)
        
        # Add levels to the dataframe
        result['support_level'] = False
        result['resistance_level'] = False
        
        # Mark candles that are at support or resistance
        for level in support_levels:
            mask = (result['low'] >= level * (1 - tolerance)) & (result['low'] <= level * (1 + tolerance))
            result.loc[mask, 'support_level'] = True
        
        for level in resistance_levels:
            mask = (result['high'] >= level * (1 - tolerance)) & (result['high'] <= level * (1 + tolerance))
            result.loc[mask, 'resistance_level'] = True
        
        # Store the actual levels
        result['support_levels'] = str(support_levels)
        result['resistance_levels'] = str(resistance_levels)
        
        return result, support_levels, resistance_levels
    
    def _cluster_levels(self, levels, tolerance):
        """
        Cluster similar price levels together.
        
        Args:
            levels (list): List of price levels
            tolerance (float): Percentage tolerance for clustering
            
        Returns:
            list: Clustered price levels
        """
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # If this level is close to the average of the current cluster
            if level <= current_cluster[-1] * (1 + tolerance):
                current_cluster.append(level)
            else:
                # Start a new cluster
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def identify_engulfing_patterns(self, df):
        """
        Identify bullish and bearish engulfing patterns.
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with engulfing patterns identified
        """
        logger.info("Identifying engulfing patterns")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate candle body size
        result['body_size'] = abs(result['close'] - result['open'])
        result['total_size'] = result['high'] - result['low']
        result['body_to_total_ratio'] = result['body_size'] / result['total_size']
        
        # Identify strong candles (at least 2/3 of the candle is body)
        result['strong_candle'] = result['body_to_total_ratio'] >= 2/3
        
        # Initialize engulfing columns
        result['bullish_engulfing'] = False
        result['bearish_engulfing'] = False
        
        # Identify engulfing patterns
        for i in range(1, len(result)):
            # Bullish engulfing
            if (result['close'].iloc[i] > result['open'].iloc[i] and  # Current candle is bullish
                result['close'].iloc[i-1] < result['open'].iloc[i-1] and  # Previous candle is bearish
                result['close'].iloc[i] > result['open'].iloc[i-1] and  # Current close > previous open
                result['open'].iloc[i] < result['close'].iloc[i-1] and  # Current open < previous close
                result['strong_candle'].iloc[i]):  # Current candle is strong
                result['bullish_engulfing'].iloc[i] = True
            
            # Bearish engulfing
            if (result['close'].iloc[i] < result['open'].iloc[i] and  # Current candle is bearish
                result['close'].iloc[i-1] > result['open'].iloc[i-1] and  # Previous candle is bullish
                result['close'].iloc[i] < result['open'].iloc[i-1] and  # Current close < previous open
                result['open'].iloc[i] > result['close'].iloc[i-1] and  # Current open > previous close
                result['strong_candle'].iloc[i]):  # Current candle is strong
                result['bearish_engulfing'].iloc[i] = True
        
        return result
    
    def identify_price_reactions(self, df, support_levels, resistance_levels, tolerance=0.001):
        """
        Identify price reactions to key levels.
        
        Args:
            df (pd.DataFrame): OHLCV data
            support_levels (list): List of support levels
            resistance_levels (list): List of resistance levels
            tolerance (float): Percentage tolerance for level proximity
            
        Returns:
            pd.DataFrame: DataFrame with price reactions identified
        """
        logger.info("Identifying price reactions")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Initialize reaction columns
        result['support_reaction'] = False
        result['resistance_reaction'] = False
        
        # Identify reactions to support levels
        for level in support_levels:
            # Price approached support and bounced up
            mask = (
                (result['low'] >= level * (1 - tolerance)) & 
                (result['low'] <= level * (1 + tolerance)) &
                (result['close'] > result['open'])  # Bullish candle
            )
            result.loc[mask, 'support_reaction'] = True
        
        # Identify reactions to resistance levels
        for level in resistance_levels:
            # Price approached resistance and bounced down
            mask = (
                (result['high'] >= level * (1 - tolerance)) & 
                (result['high'] <= level * (1 + tolerance)) &
                (result['close'] < result['open'])  # Bearish candle
            )
            result.loc[mask, 'resistance_reaction'] = True
        
        return result
    
    def generate_trade_signals(self, htf_data, ltf_data, support_levels, resistance_levels):
        """
        Generate trade signals based on HTF analysis and LTF confirmation.
        
        Args:
            htf_data (pd.DataFrame): High timeframe (H1) data with analysis
            ltf_data (pd.DataFrame): Low timeframe (M15) data with analysis
            support_levels (list): List of support levels
            resistance_levels (list): List of resistance levels
            
        Returns:
            pd.DataFrame: DataFrame with trade signals
        """
        logger.info("Generating trade signals")
        
        # Create a copy of LTF data for signals
        signals = ltf_data.copy()
        
        # Initialize signal columns
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        signals['partial_tp'] = np.nan
        signals['risk_level'] = 'low'  # Default to low risk
        
        # Get the latest HTF trend
        latest_htf_trend = htf_data['trend'].iloc[-1]
        
        # Generate signals based on LTF confirmations
        for i in range(1, len(signals)):
            # Skip if we already have an active signal
            if signals['buy_signal'].iloc[i-1] or signals['sell_signal'].iloc[i-1]:
                continue
            
            # Buy signal conditions
            if (latest_htf_trend == 'uptrend' and 
                (signals['bullish_engulfing'].iloc[i] or signals['support_reaction'].iloc[i])):
                
                signals['buy_signal'].iloc[i] = True
                
                # Set stop loss behind the order block or recent low
                if any(signals['bullish_order_block'].iloc[max(0, i-5):i]):
                    # Find the most recent bullish order block
                    for j in range(i-1, max(0, i-5), -1):
                        if signals['bullish_order_block'].iloc[j]:
                            signals['stop_loss'].iloc[i] = signals['bullish_ob_low'].iloc[j]
                            break
                else:
                    # Use recent low
                    signals['stop_loss'].iloc[i] = signals['low'].iloc[i-5:i].min()
                
                # Find the next resistance level for take profit
                next_resistance = min([r for r in resistance_levels if r > signals['close'].iloc[i]], 
                                     default=signals['close'].iloc[i] * 1.02)
                
                # Set take profit at 80% of the distance to next resistance
                distance_to_resistance = next_resistance - signals['close'].iloc[i]
                signals['take_profit'].iloc[i] = signals['close'].iloc[i] + 0.8 * distance_to_resistance
                
                # Set partial take profit at 50% of the level
                signals['partial_tp'].iloc[i] = signals['close'].iloc[i] + 0.5 * distance_to_resistance
                
                # Determine risk level based on proximity to key levels
                if signals['support_reaction'].iloc[i] or signals['bullish_order_block'].iloc[i-1:i+1].any():
                    signals['risk_level'].iloc[i] = 'low'  # 1% risk
                else:
                    signals['risk_level'].iloc[i] = 'high'  # 0.5% risk
            
            # Sell signal conditions
            elif (latest_htf_trend == 'downtrend' and 
                 (signals['bearish_engulfing'].iloc[i] or signals['resistance_reaction'].iloc[i])):
                
                signals['sell_signal'].iloc[i] = True
                
                # Set stop loss behind the order block or recent high
                if any(signals['bearish_order_block'].iloc[max(0, i-5):i]):
                    # Find the most recent bearish order block
                    for j in range(i-1, max(0, i-5), -1):
                        if signals['bearish_order_block'].iloc[j]:
                            signals['stop_loss'].iloc[i] = signals['bearish_ob_high'].iloc[j]
                            break
                else:
                    # Use recent high
                    signals['stop_loss'].iloc[i] = signals['high'].iloc[i-5:i].max()
                
                # Find the next support level for
(Content truncated due to size limit. Use line ranges to read in chunks)