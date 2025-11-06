"""
Setup script for packaging the trading bot as an executable.
This script uses PyInstaller to create a standalone executable for Windows.
"""

from setuptools import setup, find_packages

setup(
    name="trading_bot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "ccxt",
        "yfinance",
        "pandas-ta",
        "pyqt5",
        "pyinstaller",
    ],
    author="Trading Bot Developer",
    author_email="developer@example.com",
    description="Automated trading bot for Forex, Gold, and Bitcoin markets",
    keywords="trading, bot, forex, bitcoin, gold, price action",
    python_requires=">=3.8",
)
