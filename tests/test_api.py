import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "device" in data
    assert "backbone" in data

def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0
    first = data["samples"][0]
    assert "id" in first
    assert "references" in first
    assert "image_url" in first

def test_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "architecture" in data
    assert "total_parameters" in data
    assert data["total_parameters"] > 0

def test_evaluate_endpoint():
    payload = {
        "candidate": "a dog jumps to catch a frisbee",
        "references": ["a dog jumping in the air catching a red frisbee"]
    }
    response = client.post("/api/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "bleu_1" in data
    assert "bleu_4" in data
    assert "brevity_penalty" in data

def test_caption_preset_generation():
    response = client.post("/api/caption", data={
        "sample_id": "dog_frisbee",
        "strategy": "greedy"
    })
    assert response.status_code == 200
    data = response.json()
    assert "caption" in data
    assert len(data["caption"]) > 0
    assert "tokens" in data
    assert "metadata" in data
