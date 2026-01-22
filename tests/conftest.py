from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from llm_gateway.main import app


@pytest.fixture(autouse=True)
def mock_google_api_key():
    with patch(
        "llm_gateway.main.settings.GOOGLE_API_KEY",
        "fake-key",
    ):
        yield


@pytest.fixture
def app_instance():
    return app()


@pytest.fixture
def client_instance(app_instance):
    return TestClient(app_instance)


@pytest.fixture(autouse=True)
def mock_genai_client():
    with patch("llm_gateway.extensions.providers.gemini.genai.Client") as mock:
        yield mock
