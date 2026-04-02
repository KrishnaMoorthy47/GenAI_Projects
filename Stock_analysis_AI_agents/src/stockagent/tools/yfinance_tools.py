from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from crewai.tools import tool


# ── Helper functions ───────────────────────────────────────────────────────────

def _df_to_str(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def _safe(val):
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    return val


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calculate_macd(series: pd.Series, short=12, long=26, signal=9):
    macd = series.ewm(span=short, adjust=False).mean() - series.ewm(span=long, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig


def _calculate_beta(stock_returns: pd.Series, period: str) -> float:
    market = yf.Ticker("^GSPC").history(period=period)["Close"].pct_change().dropna()
    aligned = pd.concat([stock_returns, market], axis=1).dropna()
    if aligned.shape[1] < 2 or len(aligned) < 2:
        return float("nan")
    cov = aligned.cov().iloc[0, 1]
    return cov / market.var()


def _calculate_sharpe(returns: pd.Series, risk_free: float = 0.02) -> float:
    excess = returns - risk_free / 252
    return float(np.sqrt(252) * excess.mean() / excess.std()) if excess.std() != 0 else float("nan")


def _calculate_max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    return float(((prices - peak) / peak).min())


# ── CrewAI tools ───────────────────────────────────────────────────────────────

@tool
def get_basic_stock_info(ticker: str) -> str:
    """Get basic information about a stock: name, sector, market cap, current price, 52-week high/low."""
    info = yf.Ticker(ticker).info
    data = {
        "Name": info.get("longName", "N/A"),
        "Ticker": ticker.upper(),
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Current Price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
        "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        "Avg Volume": info.get("averageVolume", "N/A"),
    }
    return json.dumps({k: _safe(v) for k, v in data.items()})


@tool
def get_fundamental_analysis(ticker: str, period: str = "1y") -> str:
    """Perform fundamental analysis: PE ratio, EPS, revenue growth, margins, debt/equity."""
    info = yf.Ticker(ticker).info
    history = yf.Ticker(ticker).history(period=period)
    data = {
        "PE Ratio": info.get("trailingPE"),
        "Forward PE": info.get("forwardPE"),
        "PEG Ratio": info.get("pegRatio"),
        "Price to Book": info.get("priceToBook"),
        "Dividend Yield": info.get("dividendYield"),
        "EPS (TTM)": info.get("trailingEps"),
        "Revenue Growth": info.get("revenueGrowth"),
        "Profit Margin": info.get("profitMargins"),
        "Debt to Equity": info.get("debtToEquity"),
        "Return on Equity": info.get("returnOnEquity"),
        "Avg Price (Period)": round(history["Close"].mean(), 2) if not history.empty else "N/A",
    }
    return json.dumps({k: _safe(v) for k, v in data.items()})


@tool
def get_technical_analysis(ticker: str, period: str = "6mo") -> str:
    """Perform technical analysis: SMA50, SMA200, RSI, MACD with trend signals."""
    history = yf.Ticker(ticker).history(period=period)
    if history.empty:
        return json.dumps({"error": "No price data available"})
    history["SMA_50"] = history["Close"].rolling(50).mean()
    history["SMA_200"] = history["Close"].rolling(200).mean()
    history["RSI"] = _calculate_rsi(history["Close"])
    history["MACD"], history["Signal"] = _calculate_macd(history["Close"])
    latest = history.iloc[-1]

    def trend():
        if latest["Close"] > latest["SMA_50"] > latest["SMA_200"]:
            return "Bullish"
        if latest["Close"] < latest["SMA_50"] < latest["SMA_200"]:
            return "Bearish"
        return "Neutral"

    data = {
        "Current Price": round(float(latest["Close"]), 2),
        "SMA 50": round(float(latest["SMA_50"]), 2) if not np.isnan(latest["SMA_50"]) else "N/A",
        "SMA 200": round(float(latest["SMA_200"]), 2) if not np.isnan(latest["SMA_200"]) else "N/A",
        "RSI (14)": round(float(latest["RSI"]), 2) if not np.isnan(latest["RSI"]) else "N/A",
        "MACD": round(float(latest["MACD"]), 4),
        "MACD Signal": round(float(latest["Signal"]), 4),
        "Trend": trend(),
        "MACD Signal Direction": "Bullish" if latest["MACD"] > latest["Signal"] else "Bearish",
        "RSI Signal": "Overbought" if latest["RSI"] > 70 else ("Oversold" if latest["RSI"] < 30 else "Neutral"),
    }
    return json.dumps(data)


@tool
def get_stock_risk_assessment(ticker: str, period: str = "1y") -> str:
    """Assess stock risk: volatility, beta, VaR, Sharpe ratio, max drawdown."""
    history = yf.Ticker(ticker).history(period=period)
    if history.empty:
        return json.dumps({"error": "No price data"})
    returns = history["Close"].pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(252))
    var_95 = float(np.percentile(returns, 5))
    data = {
        "Annualized Volatility": round(volatility, 4),
        "Beta": round(_calculate_beta(returns, period), 4),
        "Value at Risk 95%": round(var_95, 4),
        "Max Drawdown": round(_calculate_max_drawdown(history["Close"]), 4),
        "Sharpe Ratio": round(_calculate_sharpe(returns), 4),
    }
    return json.dumps({k: _safe(v) for k, v in data.items()})


@tool
def get_stock_news(ticker: str, limit: int = 8) -> str:
    """Fetch recent news headlines for a stock from Yahoo Finance."""
    news = yf.Ticker(ticker).news[:limit]
    articles = []
    for a in news:
        try:
            content = a.get("content", {})
            articles.append({
                "title": content.get("title", a.get("title", "N/A")),
                "publisher": content.get("provider", {}).get("displayName", "N/A"),
                "published": content.get("pubDate", "N/A"),
                "link": content.get("canonicalUrl", {}).get("url", "N/A"),
            })
        except Exception:
            continue
    return json.dumps(articles)
