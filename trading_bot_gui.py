"""
GUI Interface for Trading Bot

This module provides a graphical user interface for the trading bot using PyQt5.
It allows users to:
- Select markets to trade
- View market analysis and charts
- Monitor open trades
- View account statistics
- Configure bot settings
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QTabWidget, QTableWidget, 
    QTableWidgetItem, QHeaderView, QSplitter, QFrame, QGridLayout,
    QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,
    QTextEdit, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon, QColor

# Add parent directory to path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_fetcher import DataFetcher
from strategy.strategy_analyzer import StrategyAnalyzer
from trade_management.trade_manager import TradeManager
from risk_management.risk_manager import RiskManager

class MarketDataThread(QThread):
    """Thread for fetching market data in the background"""
    data_updated = pyqtSignal(dict)
    
    def __init__(self, data_fetcher, symbols, timeframes):
        super().__init__()
        self.data_fetcher = data_fetcher
        self.symbols = symbols
        self.timeframes = timeframes
        self.running = True
    
    def run(self):
        while self.running:
            try:
                data = {}
                for symbol in self.symbols:
                    data[symbol] = {}
                    for timeframe in self.timeframes:
                        df = self.data_fetcher.fetch_data(symbol, timeframe)
                        data[symbol][timeframe] = df
                
                self.data_updated.emit(data)
            except Exception as e:
                print(f"Error fetching data: {e}")
            
            # Sleep for 60 seconds before next update
            for i in range(60):
                if not self.running:
                    break
                self.msleep(1000)  # Sleep for 1 second
    
    def stop(self):
        self.running = False


class ChartCanvas(FigureCanvas):
    """Canvas for displaying price charts"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_data(self, df, title="Price Chart"):
        """Plot OHLC data with indicators"""
        self.axes.clear()
        
        # Plot price data
        self.axes.plot(df.index, df['close'], label='Close Price')
        
        # Plot moving averages if available
        if 'sma_short' in df.columns:
            self.axes.plot(df.index, df['sma_short'], label='SMA Short', alpha=0.7)
        if 'sma_long' in df.columns:
            self.axes.plot(df.index, df['sma_long'], label='SMA Long', alpha=0.7)
        
        # Plot support and resistance levels if available
        if 'support_levels' in df.columns and isinstance(df['support_levels'].iloc[0], str):
            try:
                support_levels = eval(df['support_levels'].iloc[-1])
                for level in support_levels:
                    self.axes.axhline(y=level, color='g', linestyle='--', alpha=0.5)
            except:
                pass
        
        if 'resistance_levels' in df.columns and isinstance(df['resistance_levels'].iloc[0], str):
            try:
                resistance_levels = eval(df['resistance_levels'].iloc[-1])
                for level in resistance_levels:
                    self.axes.axhline(y=level, color='r', linestyle='--', alpha=0.5)
            except:
                pass
        
        # Mark buy and sell signals if available
        if 'buy_signal' in df.columns:
            buy_signals = df[df['buy_signal'] == True]
            if not buy_signals.empty:
                self.axes.scatter(buy_signals.index, buy_signals['close'], 
                                 color='green', marker='^', s=100, label='Buy Signal')
        
        if 'sell_signal' in df.columns:
            sell_signals = df[df['sell_signal'] == True]
            if not sell_signals.empty:
                self.axes.scatter(sell_signals.index, sell_signals['close'], 
                                 color='red', marker='v', s=100, label='Sell Signal')
        
        # Format the chart
        self.axes.set_title(title)
        self.axes.set_xlabel('Date')
        self.axes.set_ylabel('Price')
        self.axes.legend(loc='best')
        self.axes.grid(True, alpha=0.3)
        
        # Format x-axis dates
        self.fig.autofmt_xdate()
        
        self.draw()


class TradesTableWidget(QTableWidget):
    """Table widget for displaying trades"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(9)
        self.setHorizontalHeaderLabels([
            'ID', 'Symbol', 'Type', 'Entry Price', 'Current Price', 
            'Stop Loss', 'Take Profit', 'P&L', 'Risk'
        ])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
    
    def update_trades(self, trades):
        """Update the table with current trades"""
        self.setRowCount(0)
        
        for trade in trades:
            row_position = self.rowCount()
            self.insertRow(row_position)
            
            # Set trade data
            self.setItem(row_position, 0, QTableWidgetItem(str(trade.get('id', ''))))
            self.setItem(row_position, 1, QTableWidgetItem(trade.get('symbol', '')))
            self.setItem(row_position, 2, QTableWidgetItem(trade.get('type', '').upper()))
            self.setItem(row_position, 3, QTableWidgetItem(f"{trade.get('entry_price', 0):.2f}"))
            self.setItem(row_position, 4, QTableWidgetItem(f"{trade.get('current_price', 0):.2f}"))
            self.setItem(row_position, 5, QTableWidgetItem(f"{trade.get('stop_loss', 0):.2f}"))
            self.setItem(row_position, 6, QTableWidgetItem(f"{trade.get('take_profit', 0):.2f}"))
            
            # Format P&L with color
            pnl = trade.get('pnl', 0)
            pnl_item = QTableWidgetItem(f"${pnl:.2f}")
            if pnl > 0:
                pnl_item.setForeground(QColor('green'))
            elif pnl < 0:
                pnl_item.setForeground(QColor('red'))
            self.setItem(row_position, 7, pnl_item)
            
            # Risk level
            risk_percentage = trade.get('risk_percentage', 0)
            risk_item = QTableWidgetItem(f"{risk_percentage:.1f}%")
            self.setItem(row_position, 8, risk_item)


class TradingBotGUI(QMainWindow):
    """Main GUI window for the trading bot"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_storage')
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.data_fetcher = DataFetcher(data_dir=self.data_dir)
        self.strategy_analyzer = StrategyAnalyzer()
        self.trade_manager = TradeManager(account_balance=10000, trades_file=os.path.join(self.data_dir, 'trades.json'))
        self.risk_manager = RiskManager(account_balance=10000, max_open_trades=2)
        
        # Available markets and timeframes
        self.symbols = ['BTCUSD', 'XAUUSD', 'EURUSD']
        self.timeframes = ['H1', 'M15']
        
        # Current selections
        self.current_symbol = 'BTCUSD'
        self.current_timeframe = 'H1'
        
        # Data storage
        self.market_data = {}
        self.analysis_data = {}
        
        # Initialize UI
        self.init_ui()
        
        # Start data thread
        self.data_thread = MarketDataThread(self.data_fetcher, self.symbols, self.timeframes)
        self.data_thread.data_updated.connect(self.on_data_updated)
        self.data_thread.start()
        
        # Update timer for UI refresh (every 5 seconds)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(5000)
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Trading Bot')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create header with controls
        header_layout = QHBoxLayout()
        
        # Market selection
        market_label = QLabel('Market:')
        self.market_combo = QComboBox()
        self.market_combo.addItems(self.symbols)
        self.market_combo.currentTextChanged.connect(self.on_market_changed)
        
        # Timeframe selection
        timeframe_label = QLabel('Timeframe:')
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(self.timeframes)
        self.timeframe_combo.currentTextChanged.connect(self.on_timeframe_changed)
        
        # Refresh button
        refresh_button = QPushButton('Refresh Data')
        refresh_button.clicked.connect(self.refresh_data)
        
        # Add widgets to header layout
        header_layout.addWidget(market_label)
        header_layout.addWidget(self.market_combo)
        header_layout.addWidget(timeframe_label)
        header_layout.addWidget(self.timeframe_combo)
        header_layout.addWidget(refresh_button)
        header_layout.addStretch()
        
        # Create tab widget for different sections
        self.tabs = QTabWidget()
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_charts_tab()
        self.create_trades_tab()
        self.create_settings_tab()
        
        # Add layouts to main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage('Ready')
    
    def create_dashboard_tab(self):
        """Create the dashboard tab"""
        dashboard_widget = QWidget()
        layout = QVBoxLayout(dashboard_widget)
        
        # Account summary section
        account_group = QGroupBox('Account Summary')
        account_layout = QGridLayout()
        
        self.balance_label = QLabel('Balance: $10,000.00')
        self.open_trades_label = QLabel('Open Trades: 0')
        self.total_pnl_label = QLabel('Total P&L: $0.00')
        self.win_rate_label = QLabel('Win Rate: 0%')
        
        account_layout.addWidget(self.balance_label, 0, 0)
        account_layout.addWidget(self.open_trades_label, 0, 1)
        account_layout.addWidget(self.total_pnl_label, 1, 0)
        account_layout.addWidget(self.win_rate_label, 1, 1)
        
        account_group.setLayout(account_layout)
        
        # Market overview section
        market_group = QGroupBox('Market Overview')
        market_layout = QVBoxLayout()
        
        self.market_status_text = QTextEdit()
        self.market_status_text.setReadOnly(True)
        
        market_layout.addWidget(self.market_status_text)
        market_group.setLayout(market_layout)
        
        # Recent signals section
        signals_group = QGroupBox('Recent Signals')
        signals_layout = QVBoxLayout()
        
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(5)
        self.signals_table.setHorizontalHeaderLabels([
            'Symbol', 'Timeframe', 'Type', 'Price', 'Time'
        ])
        self.signals_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        signals_layout.addWidget(self.signals_table)
        signals_group.setLayout(signals_layout)
        
        # Add all sections to layout
        layout.addWidget(account_group)
        layout.addWidget(market_group)
        layout.addWidget(signals_group)
        
        self.tabs.addTab(dashboard_widget, 'Dashboard')
    
    def create_charts_tab(self):
        """Create the charts tab"""
        charts_widget = QWidget()
        layout = QVBoxLayout(charts_widget)
        
        # Create chart canvases
        self.htf_chart = ChartCanvas(width=5, height=4)
        self.ltf_chart = ChartCanvas(width=5, height=4)
        
        # Add charts to layout
        layout.addWidget(QLabel('High Timeframe (H1)'))
        layout.addWidget(self.htf_chart)
        layout.addWidget(QLabel('Low Timeframe (M15)'))
        layout.addWidget(self.ltf_chart)
        
        self.tabs.addTab(charts_widget, 'Charts')
    
    def create_trades_tab(self):
        """Create the trades tab"""
        trades_widget = QWidget()
        layout = QVBoxLayout(trades_widget)
        
        # Open trades section
        open_trades_group = QGroupBox('Open Trades')
        open_trades_layout = QVBoxLayout()
        
        self.open_trades_table = TradesTableWidget()
        open_trades_layout.addWidget(self.open_trades_table)
        open_trades_group.setLayout(open_trades_layout)
        
        # Trade history section
        history_group = QGroupBox('Trade History')
        history_layout = QVBoxLayout()
        
        self.trade_history_table = QTableWidget()
        self.trade_history_table.setColumnCount(8)
        self.trade_history_table.setHorizontalHeaderLabels([
            'ID', 'Symbol', 'Type', 'Entry Price', 'Exit Price', 
            'Exit Reason', 'P&L', 'Duration'
        ])
        self.trade_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        history_layout.addWidget(self.trade_history_table)
        history_group.setLayout(history_layout)
        
        # Add sections to layout
        layout.addWidget(open_trades_group)
        layout.addWidget(history_group)
        
        self.tabs.addTab(trades_widget, 'Trades')
    
    def create_settings_tab(self):
        """Create the settings tab"""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # Trading settings
        trading_group = QGroupBox('Trading Settings')
        trading_layout = QFormLayout()
        
        self.account_balance_spin = QDoubleSpinBox()
        self.account_balance_spin.setRange(100, 1000000)
        self.account_balance_spin.setValue(10000)
        self.account_balance_spin.setPrefix('$ ')
        self.account_balance_spin.setDecimals(2)
        
        self.max_trades_spin = QSpinBox()
        self.max_trades_spin.setRange(1, 10)
        self.max_trades_spin.setValue(2)
        
        self.high_risk_spin = QDoubleSpinBox()
        self.high_risk_spin.setRange(0.1, 5)
        self.high_risk_spin.setValue(0.5)
        self.high_risk_spin.setSuffix(' %')
        self.high_risk_spin.setSingleStep(0.1)
        
        self.low_risk_spin = QDoubleSpinBox()
        self.low_risk_spin.setRange(0.1, 5)
        self.low_risk_spin.setValue(1.0)
        self.low_risk_spin.setSuffix(' %')
        self.low_risk_spin.setSingleStep(0.1)
        
        trading_layout.addRow('Account Balance:', self.account_balance_spin)
        trading_layout.addRow('Max Open Trades:', self.max_trades_spin)
        trading_layout.addRow('High Risk (%):', self.high_risk_spin)
        trading_layout.addRow('Low Risk (%):', self.low_risk_spin)
        
        trading_group.setLayout(trading_layout)
        
        # Market settings
        market_group = QGroupBox('Market Settings')
        market_layout = QVBoxLayout()
        
        # Checkboxes for markets
        self.market_chec
(Content truncated due to size limit. Use line ranges to read in chunks)