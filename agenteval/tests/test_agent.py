"""Unit tests for agent nodes and tools."""

import sqlite3

import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agenteval.agent import tools as tools_module
from agenteval.agent.nodes import make_agent_node
from agenteval.agent.state import SQLAgentState
from tests.conftest import make_mock_llm


# ──────────────────────────────────────────────
# Tool tests (using in-memory DB)
# ──────────────────────────────────────────────

class TestTools:
    def setup_method(self):
        """Point tools at the test DB before each test."""
        pass

    def test_list_tables(self, mock_db):
        tools_module.set_db_path(mock_db)
        result = tools_module.list_tables.invoke({})
        assert "Song" in result
        assert "Composer" in result
        assert "Singer" in result

    def test_get_schema(self, mock_db):
        tools_module.set_db_path(mock_db)
        result = tools_module.get_schema.invoke({"table_name": "Song"})
        assert "SongId" in result
        assert "GenreId" in result

    def test_get_schema_missing_table(self, mock_db):
        tools_module.set_db_path(mock_db)
        result = tools_module.get_schema.invoke({"table_name": "NonExistent"})
        assert "not found" in result

    def test_run_query_select(self, mock_db):
        tools_module.set_db_path(mock_db)
        result = tools_module.run_query.invoke({"query": "SELECT COUNT(*) FROM Song"})
        assert "6" in result  # 6 songs in test data

    def test_run_query_blocks_non_select(self, mock_db):
        tools_module.set_db_path(mock_db)
        result = tools_module.run_query.invoke({"query": "DELETE FROM Song"})
        assert "Only SELECT" in result

    def test_run_query_invalid_sql(self, mock_db):
        tools_module.set_db_path(mock_db)
        result = tools_module.run_query.invoke({"query": "SELECT * FROM NonExistentTable123"})
        assert "SQL error" in result or "error" in result.lower()

    def test_run_query_no_results(self, mock_db):
        tools_module.set_db_path(mock_db)
        result = tools_module.run_query.invoke({"query": "SELECT * FROM Genre WHERE Name='ZZZ'"})
        assert "No results" in result


# ──────────────────────────────────────────────
# Agent node tests
# ──────────────────────────────────────────────

class TestAgentNode:
    def _make_state(self, **kwargs) -> SQLAgentState:
        base: SQLAgentState = {
            "question": "How many songs?",
            "messages": [
                SystemMessage(content="You are a SQL agent."),
                HumanMessage(content="How many songs?"),
            ],
            "trajectory": [],
            "sql_result": None,
            "answer": None,
            "iterations": 0,
        }
        base.update(kwargs)
        return base

    def test_agent_node_adds_tool_calls_to_trajectory(self):
        """When LLM returns tool calls, trajectory should be updated."""
        mock_llm = make_mock_llm()
        agent_node = make_agent_node(mock_llm)

        state = self._make_state()
        result = agent_node(state)

        # First call should have list_tables tool call
        assert "trajectory" in result
        assert len(result["trajectory"]) >= 1

    def test_agent_node_max_iterations(self):
        """At max iterations, should set answer and not loop further."""
        from agenteval.config import get_settings
        max_iter = get_settings().max_agent_iterations

        mock_llm = make_mock_llm()
        agent_node = make_agent_node(mock_llm)

        # Simulate being at the iteration limit
        state = self._make_state(
            iterations=max_iter,
            messages=[
                SystemMessage(content="sys"),
                HumanMessage(content="q"),
                AIMessage(content="I give up."),
            ],
        )
        result = agent_node(state)
        assert result.get("answer") is not None

    def test_agent_node_extracts_answer_when_no_tool_calls(self):
        """When LLM returns no tool calls, answer should be extracted."""
        final_answer_msg = AIMessage(content="There are 6 songs.")
        mock_llm = make_mock_llm(tool_calls_sequence=[final_answer_msg])
        agent_node = make_agent_node(mock_llm)

        state = self._make_state()
        result = agent_node(state)
        assert result.get("answer") == "There are 6 songs."


# ──────────────────────────────────────────────
# Graph integration test
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_sql_agent_end_to_end(mock_db):
    """Full graph run with mocked LLM and test DB."""
    from agenteval.agent import tools as tools_module
    from agenteval.agent.graph import run_sql_agent

    tools_module.set_db_path(mock_db)

    # LLM that produces: list_tables → run_query → final answer
    final_answer = "There are 6 songs in the database."
    mock_llm = make_mock_llm(final_answer=final_answer)

    state = await run_sql_agent("How many songs are there?", llm=mock_llm)

    assert state["answer"] is not None
    assert state["question"] == "How many songs are there?"
