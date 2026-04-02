from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def mock_llm():
    """Fixture returning a deterministic mock LLM."""
    llm = MagicMock()
    llm.invoke = MagicMock(
        return_value=AIMessage(content="No security issues found in this diff.")
    )
    llm.bind_tools = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def sample_diff() -> str:
    return '''diff --git a/app/api/users.py b/app/api/users.py
index abc1234..def5678 100644
--- a/app/api/users.py
+++ b/app/api/users.py
@@ -1,5 +1,15 @@
+import os
+import sqlite3
+
+def get_user(user_id: str):
+    conn = sqlite3.connect("users.db")
+    cursor = conn.cursor()
+    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
+    return cursor.fetchone()
+
+def authenticate(username: str, password: str):
+    db_password = "hardcoded_secret_123"
+    if password == db_password:
+        return True
+    return False
'''


@pytest.fixture
def clean_diff() -> str:
    return '''diff --git a/app/utils.py b/app/utils.py
index abc1234..def5678 100644
--- a/app/utils.py
+++ b/app/utils.py
@@ -1,3 +1,8 @@
+def format_name(first: str, last: str) -> str:
+    """Format a full name from first and last components."""
+    if not first or not last:
+        raise ValueError("Both first and last name are required")
+    return f"{first.strip()} {last.strip()}"
'''
