def test_health_check(client_instance):
    response = client_instance.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root(client_instance):
    response = client_instance.get("/")
    assert response.status_code == 200
    assert "LLM Gateway is running" in response.json()["message"]
