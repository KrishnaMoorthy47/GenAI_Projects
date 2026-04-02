from __future__ import annotations

import json
from typing import Optional

import httpx
import yfinance as yf
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_ticker_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info or {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_earnings_history(ticker: str):
    return yf.Ticker(ticker).earnings_history


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_calendar(ticker: str):
    return yf.Ticker(ticker).calendar


@tool
def get_stock_info(ticker: str) -> str:
    """Get current stock price, market cap, P/E ratio, 52-week range, and key fundamentals.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT, GOOGL)
    """
    try:
        info = _fetch_ticker_info(ticker.upper())

        result = {
            "ticker": ticker.upper(),
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "revenue_ttm": info.get("totalRevenue"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "free_cashflow": info.get("freeCashflow"),
            "analyst_rating": info.get("recommendationKey"),
            "target_price": info.get("targetMeanPrice"),
            "description": (info.get("longBusinessSummary") or "")[:500],
        }
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": f"Failed to fetch stock info for {ticker}: {exc}"})


@tool
def get_earnings_history(ticker: str) -> str:
    """Get the last 4 quarters of earnings data including EPS actual vs estimate.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT, GOOGL)
    """
    try:
        # Quarterly earnings history
        earnings_hist = _fetch_earnings_history(ticker.upper())
        if earnings_hist is None or earnings_hist.empty:
            quarters = []
        else:
            quarters = []
            for _, row in earnings_hist.head(4).iterrows():
                quarters.append({
                    "date": str(row.name) if hasattr(row, "name") else "N/A",
                    "eps_estimate": row.get("epsEstimate"),
                    "eps_actual": row.get("epsActual"),
                    "eps_surprise_pct": row.get("epsDifference"),
                })

        # Next earnings date
        calendar = _fetch_calendar(ticker.upper())
        next_earnings = None
        if calendar is not None and not calendar.empty:
            if "Earnings Date" in calendar.index:
                next_earnings = str(calendar.loc["Earnings Date"].iloc[0])

        return json.dumps({
            "ticker": ticker.upper(),
            "earnings_history": quarters,
            "next_earnings_date": next_earnings,
        }, default=str)
    except Exception as exc:
        return json.dumps({"error": f"Failed to fetch earnings for {ticker}: {exc}"})


_SEC_HEADERS = {"User-Agent": "FinAgent portfolio-project contact@example.com"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_sec_tickers() -> dict:
    with httpx.Client(timeout=15) as client:
        resp = client.get("https://www.sec.gov/files/company_tickers.json", headers=_SEC_HEADERS)
        resp.raise_for_status()
        return resp.json()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_sec_submissions(cik: str) -> dict:
    with httpx.Client(timeout=15) as client:
        resp = client.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=_SEC_HEADERS)
        resp.raise_for_status()
        return resp.json()


@tool
def get_sec_filings(ticker: str, filing_type: str = "10-K") -> str:
    """Get recent SEC filings metadata for a company from SEC EDGAR.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, MSFT, GOOGL)
        filing_type: SEC filing type — '10-K' (annual), '10-Q' (quarterly), or '8-K' (current report)
    """
    try:
        tickers_data = _fetch_sec_tickers()

        cik: Optional[str] = None
        for entry in tickers_data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                break

        if not cik:
            return json.dumps({"error": f"CIK not found for ticker {ticker}"})

        submissions = _fetch_sec_submissions(cik)

        filings = submissions.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accession_numbers = filings.get("accessionNumber", [])
        descriptions = filings.get("primaryDocument", [])

        results = []
        for form, date, accession, doc in zip(forms, dates, accession_numbers, descriptions):
            if form == filing_type:
                acc_clean = accession.replace("-", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
                    f"/{acc_clean}/{doc}"
                )
                results.append({
                    "form": form,
                    "filing_date": date,
                    "accession_number": accession,
                    "url": filing_url,
                })
                if len(results) >= 3:
                    break

        return json.dumps({
            "ticker": ticker.upper(),
            "cik": cik,
            "filing_type": filing_type,
            "filings": results,
        }, default=str)
    except Exception as exc:
        return json.dumps({"error": f"Failed to fetch SEC filings for {ticker}: {exc}"})
