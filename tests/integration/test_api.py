from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cifar_classifier.api import create_app
from cifar_classifier.config import Settings
from tests.conftest import png_bytes


def _client(model_path: Path, **overrides) -> TestClient:
    settings = Settings(_env_file=None, model_path=model_path, device="cpu", **overrides)
    return TestClient(create_app(settings))


@pytest.fixture
def client(model_path: Path):
    with _client(
        model_path / "resnet.pt" if model_path.is_dir() else model_path,
        max_batch_size=3,
        max_upload_bytes=5_000,
    ) as c:
        yield c


def _file(name: str = "cat.png", data: bytes | None = None):
    return ("file", (name, data if data is not None else png_bytes(), "image/png"))


def test_root_redirects(client: TestClient):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_and_info(client: TestClient):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "device": "cpu"}
    info = client.get("/model/info").json()
    assert info["model"] == "resnet9"
    assert len(info["class_names"]) == 10


def test_predict(client: TestClient):
    response = client.post("/predict", files=[_file()])
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["label"] <= 9
    assert len(body["top_k"]) == 3


def test_predict_resizes_large_jpeg(client: TestClient):
    response = client.post("/predict", files=[_file("big.jpg", png_bytes((200, 150), fmt="JPEG"))])
    assert response.status_code == 200


@pytest.mark.parametrize("data", [b"not an image", b""])
def test_predict_rejects_non_images(client: TestClient, data: bytes):
    response = client.post("/predict", files=[_file("x.png", data)])
    assert response.status_code == 422


def test_predict_requires_file(client: TestClient):
    assert client.post("/predict").status_code == 422
    assert client.post("/predict", data={"file": "text"}).status_code == 422


def test_predict_rejects_large_upload(client: TestClient):
    response = client.post("/predict", files=[_file("huge.png", b"\x89PNG" + b"0" * 6_000)])
    assert response.status_code == 413


def test_batch(client: TestClient):
    files = [("files", (f"img{i}.png", png_bytes(), "image/png")) for i in range(3)]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert [p["filename"] for p in predictions] == ["img0.png", "img1.png", "img2.png"]
    single = client.post("/predict", files=[_file()]).json()
    assert predictions[0]["label"] == single["label"]


def test_batch_too_large(client: TestClient):
    files = [("files", (f"img{i}.png", png_bytes(), "image/png")) for i in range(4)]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 413
    assert "limit of 3" in response.json()["detail"]


def test_batch_with_invalid_image(client: TestClient):
    files = [
        ("files", ("ok.png", png_bytes(), "image/png")),
        ("files", ("bad.png", b"x", "image/png")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 422
    assert "bad.png" in response.json()["detail"]


def test_batch_requires_files(client: TestClient):
    assert client.post("/predict/batch").status_code == 422


def test_without_model(tmp_path: Path):
    with _client(tmp_path / "missing.pt") as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/model/info").status_code == 503
        assert c.post("/predict", files=[_file()]).status_code == 503
        files = [("files", ("a.png", png_bytes(), "image/png"))]
        assert c.post("/predict/batch", files=files).status_code == 503


def test_corrupt_model(tmp_path: Path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"garbage")
    with _client(path) as c:
        assert c.post("/predict", files=[_file()]).status_code == 503
