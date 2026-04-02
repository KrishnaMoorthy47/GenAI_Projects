from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestCalculator:
    def test_basic_arithmetic(self):
        from finagent.tools.calculator import calculator
        assert calculator.invoke({"expression": "2 + 2"}) == "4.0"

    def test_percentage_calculation(self):
        from finagent.tools.calculator import calculator
        result = calculator.invoke({"expression": "(150 - 120) / 120 * 100"})
        assert result == "25.0"

    def test_sqrt(self):
        from finagent.tools.calculator import calculator
        result = calculator.invoke({"expression": "sqrt(144)"})
        assert result == "12.0"

    def test_division_by_zero(self):
        from finagent.tools.calculator import calculator
        result = calculator.invoke({"expression": "1 / 0"})
        assert "Division by zero" in result

    def test_unsafe_expression_rejected(self):
        from finagent.tools.calculator import calculator
        result = calculator.invoke({"expression": "__import__('os').system('ls')"})
        assert "Error" in result

    def test_power_operator(self):
        from finagent.tools.calculator import calculator
        result = calculator.invoke({"expression": "2 ** 10"})
        assert result == "1024.0"


class TestFinanceTools:
    def test_get_stock_info_returns_json(self):
        from finagent.tools.finance import get_stock_info

        mock_info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "currentPrice": 175.0,
            "marketCap": 2_700_000_000_000,
            "trailingPE": 28.5,
        }
        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("finagent.tools.finance.yf.Ticker", return_value=mock_ticker):
            result = get_stock_info.invoke({"ticker": "AAPL"})

        data = json.loads(result)
        assert data["ticker"] == "AAPL"
        assert data["name"] == "Apple Inc."
        assert data["sector"] == "Technology"
        assert data["current_price"] == 175.0

    def test_get_stock_info_handles_error(self):
        from finagent.tools.finance import get_stock_info

        with patch("finagent.tools.finance.yf.Ticker", side_effect=Exception("Network error")):
            result = get_stock_info.invoke({"ticker": "INVALID"})

        data = json.loads(result)
        assert "error" in data

    def test_get_earnings_history_empty(self):
        from finagent.tools.finance import get_earnings_history

        mock_ticker = MagicMock()
        mock_ticker.earnings_history = None
        mock_ticker.calendar = None

        with patch("finagent.tools.finance.yf.Ticker", return_value=mock_ticker):
            result = get_earnings_history.invoke({"ticker": "AAPL"})

        data = json.loads(result)
        assert data["ticker"] == "AAPL"
        assert data["earnings_history"] == []


class TestWebSearchTool:
    def test_web_search_formats_results(self):
        from finagent.tools.web_search import web_search

        mock_results = [
            {"title": "Apple Q4 Results", "url": "https://example.com/1", "content": "Apple reported record revenue..."},
            {"title": "AAPL Analysis", "url": "https://example.com/2", "content": "Analysts are bullish on Apple..."},
        ]

        with patch("finagent.tools.web_search._get_search_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.invoke = MagicMock(return_value=mock_results)
            mock_client_fn.return_value = mock_client

            result = web_search.invoke({"query": "AAPL earnings Q4 2024"})

        assert "Apple Q4 Results" in result
        assert "AAPL Analysis" in result

    def test_web_search_handles_error(self):
        from finagent.tools.web_search import web_search

        with patch("finagent.tools.web_search._get_search_client", side_effect=Exception("API error")):
            result = web_search.invoke({"query": "test query"})

        assert "failed" in result.lower() or "error" in result.lower()
