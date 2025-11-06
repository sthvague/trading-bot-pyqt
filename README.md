# 💹 Trading Bot (Python + PyQt5)

### 🧠 Overview
This project is an **automated trading bot** designed for **Forex, Gold, and Bitcoin markets**.  
It applies **price action strategies** (market structure, order blocks, key levels, engulfing patterns) to generate trade signals, manage risk, and visualize data through a **PyQt5 graphical interface**.

The bot is modular, with separate components for data fetching, strategy analysis, trade management, risk management, and backtesting.

---

## ⚙️ Features
- 📊 **Market Data Fetching** via Yahoo Finance (`yfinance`)
- 🧩 **Price Action Strategy Analyzer**
  - Detects trends, order blocks, support/resistance, and engulfing patterns  
- 💰 **Trade Manager**
  - Executes trades based on signals and manages open/closed positions  
- ⚠️ **Risk Manager**
  - Calculates position sizes and enforces risk limits  
- ⏪ **Backtesting Engine**
  - Simulates historical trades and performance  
- 🖥️ **GUI (PyQt5)**
  - Real-time dashboard, charts, trade monitoring, and bot controls  

---

## 🧱 Project Structure
```
trading-bot/
├── backtester.py           # Backtesting module
├── data_fetcher.py         # Market data fetching
├── package.py              # Packaging helper
├── risk_manager.py         # Risk management logic
├── strategy_analyzer.py    # Strategy and signal generation
├── trade_manager.py        # Trade execution and tracking
├── trading_bot_gui.py      # Main GUI (PyQt5)
├── test_trading_bot.py     # Unit tests for all modules
├── setup.py                # Packaging setup (PyInstaller)
└── requirements.txt        # Dependencies (recommended)
```

---

## 🚀 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOURUSERNAME/trading-bot-pyqt.git
cd trading-bot-pyqt
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
If you don’t have a `requirements.txt`, use:
```bash
pip install pandas numpy matplotlib ccxt yfinance pandas-ta pyqt5 pyinstaller
```

---

## ▶️ Running the Bot (GUI)
```bash
python trading_bot_gui.py
```
This will launch the **Graphical User Interface**, allowing you to:
- Select a market (BTC/USD, XAU/USD, EUR/USD)
- View charts and signals
- Monitor open trades and account statistics
- Configure trading parameters

---

## ⏪ Running a Backtest
You can backtest strategies using:
```bash
python backtester.py
```
Results (PNL, win rate, drawdown, etc.) are printed in the console or saved to a file.

---

## 🧩 How It Works

| Module | Description |
|--------|--------------|
| `data_fetcher.py` | Fetches OHLCV data from Yahoo Finance or exchanges |
| `strategy_analyzer.py` | Analyzes price data and generates buy/sell signals |
| `trade_manager.py` | Opens, updates, and closes trades based on signals |
| `risk_manager.py` | Controls risk exposure and validates trades |
| `backtester.py` | Runs historical tests to evaluate strategy performance |
| `trading_bot_gui.py` | Provides an interactive GUI dashboard for users |

---

## 🛠️ Build Executable (Optional)
To package the bot into a standalone `.exe`:
```bash
pyinstaller --onefile --windowed trading_bot_gui.py
```
The executable will appear in the `dist/` folder.

---

## 🧾 License
MIT License (free for personal and commercial use)
