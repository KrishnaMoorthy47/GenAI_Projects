from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver


class TestGraphCompilation:
    def test_graph_compiles_with_mock_checkpointer(self):
        """Graph should compile without errors when given a checkpointer."""
        from finagent.agents.graph import build_graph

        graph = build_graph(MemorySaver())
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """Verify all expected nodes are registered in the graph."""
        from finagent.agents.graph import build_graph

        graph = build_graph(MemorySaver())

        node_names = set(graph.nodes.keys())
        expected = {"supervisor", "web_research", "financial_data", "sentiment", "report_writer", "human_review"}
        assert expected.issubset(node_names)


class TestSupervisorRouting:
    def test_routes_to_web_research_first(self):
        """Supervisor should route to web_research when no research is done."""
        from finagent.agents.supervisor import supervisor_node

        state = {
            "ticker": "AAPL",
            "query": "Analyze AAPL",
            "messages": [],
            "web_research": None,
            "financial_data": None,
            "sentiment": None,
            "report": None,
            "next_agent": "",
            "human_approved": False,
            "thread_id": "test-thread",
        }

        command = supervisor_node(state)
        assert command.goto == "web_research"

    def test_routes_to_financial_data_after_web(self):
        """After web_research is done, supervisor should route to financial_data."""
        from finagent.agents.supervisor import supervisor_node

        state = {
            "ticker": "AAPL",
            "query": "Analyze AAPL",
            "messages": [],
            "web_research": {"summary": "Web research complete"},
            "financial_data": None,
            "sentiment": None,
            "report": None,
            "next_agent": "",
            "human_approved": False,
            "thread_id": "test-thread",
        }

        command = supervisor_node(state)
        assert command.goto == "financial_data"

    def test_routes_to_sentiment_after_both_research(self):
        """After web + financial data, supervisor should route to sentiment."""
        from finagent.agents.supervisor import supervisor_node

        state = {
            "ticker": "AAPL",
            "query": "Analyze AAPL",
            "messages": [],
            "web_research": {"summary": "done"},
            "financial_data": {"summary": "done"},
            "sentiment": None,
            "report": None,
            "next_agent": "",
            "human_approved": False,
            "thread_id": "test-thread",
        }

        command = supervisor_node(state)
        assert command.goto == "sentiment"

    def test_routes_to_report_writer_when_all_done(self):
        """Supervisor should route to report_writer when all research is complete."""
        from finagent.agents.supervisor import supervisor_node

        state = {
            "ticker": "AAPL",
            "query": "Analyze AAPL",
            "messages": [],
            "web_research": {"summary": "done"},
            "financial_data": {"summary": "done"},
            "sentiment": {"summary": "done"},
            "report": None,
            "next_agent": "",
            "human_approved": False,
            "thread_id": "test-thread",
        }

        command = supervisor_node(state)
        assert command.goto == "report_writer"


class TestStreamingService:
    def test_create_and_get_queue(self):
        from finagent.services.streaming import create_stream_queue, get_stream_queue, get_status

        thread_id = "test-stream-123"
        q = create_stream_queue(thread_id)

        assert q is not None
        assert get_stream_queue(thread_id) is q
        assert get_status(thread_id) == "running"

    @pytest.mark.asyncio
    async def test_sse_generator_returns_error_for_unknown_thread(self):
        from finagent.services.streaming import sse_generator

        events = []
        async for event in sse_generator("nonexistent-thread"):
            events.append(event)
            break

        assert len(events) == 1
        assert "error" in events[0].get("data", "")
