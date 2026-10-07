import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.inference.music_generator import MusicGenerator


@pytest.fixture
def client():
    return TestClient(app)


def test_import_inference_module():
    """Verify MusicGenerator class can be imported and initialized without errors."""
    generator = MusicGenerator(
        base_model_name="facebook/musicgen-small",
        adapter_path="models/adapters/adapter3",
    )
    assert generator.base_model_name == "facebook/musicgen-small"
    assert str(generator.adapter_path).replace("\\", "/") == "models/adapters/adapter3"


def test_health_endpoint(client):
    """Verify GET /health returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_endpoint_empty_prompt(client):
    """Verify POST /generate rejects empty prompt with 400 Bad Request."""
    response = client.post("/generate", json={"prompt": "   "})
    assert response.status_code == 400
    assert "Prompt cannot be empty" in response.json()["detail"]
