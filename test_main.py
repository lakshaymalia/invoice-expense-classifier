from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"text": "AWS monthly cloud hosting bill"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert data["category"] == "Cloud/Software"