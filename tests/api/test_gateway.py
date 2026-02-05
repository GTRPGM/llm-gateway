import pytest
from fastapi.testclient import TestClient

from llm_gateway.main import app


@pytest.fixture
def client():
    return TestClient(app())


def test_get_gateway_config(client):
    response = client.get("/api/v1/gateway/config")
    assert response.status_code == 200
    data = response.json()
    assert "default_provider" in data
    assert data["default_provider"] in ["google", "openai"]


def test_update_gateway_config_success(client):
    # Test switching to openai
    response = client.post(
        "/api/v1/gateway/config", json={"default_provider": "openai"}
    )
    assert response.status_code == 200
    assert response.json()["default_provider"] == "openai"

    # Test switching back to google
    response = client.post(
        "/api/v1/gateway/config", json={"default_provider": "google"}
    )
    assert response.status_code == 200
    assert response.json()["default_provider"] == "google"


def test_update_gateway_config_invalid_provider(client):
    response = client.post(
        "/api/v1/gateway/config", json={"default_provider": "invalid"}
    )
    assert response.status_code == 400
    assert "Provider invalid not available" in response.json()["detail"]
