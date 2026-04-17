"""Fixtures and mocks for tests."""
import os
from unittest.mock import MagicMock, patch

import pytest


# Set a dummy OpenAI API key for tests
@pytest.fixture(scope="session", autouse=True)
def mock_openai_env():
    """Set dummy OpenAI API key to avoid validation errors."""
    os.environ["OPENAI_API_KEY"] = "test-key-123456"
    yield
    if os.environ.get("OPENAI_API_KEY") == "test-key-123456":
        del os.environ["OPENAI_API_KEY"]


@pytest.fixture
def mock_chatgpt():
    """Mock ChatOpenAI to avoid actual API calls."""
    with patch("audit_rag.generator.ChatOpenAI") as mock:
        instance = MagicMock()
        instance.invoke.return_value = MagicMock(content="Test response")
        mock.return_value = instance
        yield instance
