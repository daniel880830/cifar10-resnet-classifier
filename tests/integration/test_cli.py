import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cifar_classifier import data
from cifar_classifier.cli import app
from cifar_classifier.config import get_settings
from cifar_classifier.data import FILES, file_digest

runner = CliRunner()


def invoke(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result


def test_train_single_experiment(env: Path):
    result = invoke("train", "--experiment", "resnet")
    assert "resnet" in result.output
    assert (env / "models" / "resnet.pt").is_file()
    assert (env / "reports" / "resnet.json").is_file()
    assert (env / "reports" / "figures" / "confusion_resnet.png").is_file()


def test_train_on_validation_does_not_save(env: Path):
    invoke("train", "--experiment", "resnet", "--eval-on", "val")
    assert (env / "reports" / "resnet_val.json").is_file()
    assert not (env / "models").exists()


def test_train_seeds(env: Path):
    result = invoke("train", "--experiment", "resnet", "--seeds", "2")
    assert "2 seeds" in result.output
    summary = json.loads((env / "reports" / "resnet_seeds.json").read_text())
    assert len(summary["runs"]) == 2


def test_train_rejects_bad_eval_on(env: Path):
    assert (
        runner.invoke(app, ["train", "--experiment", "resnet", "--eval-on", "test2"]).exit_code != 0
    )


def test_sweep(env: Path):
    result = invoke("sweep", "--experiment", "resnet", "--param", "lr=0.01,0.05")
    assert "best:" in result.output
    assert (env / "reports" / "resnet_sweep_val.json").is_file()


def test_download_and_verify_data(env: Path, monkeypatch: pytest.MonkeyPatch):
    import hashlib

    from tests.conftest import make_archive, make_arrays

    source = make_archive(env / "src.tar.gz", make_arrays(5, 0), make_arrays(2, 1))
    monkeypatch.setattr(data, "CIFAR10_MD5", hashlib.md5(source.read_bytes()).hexdigest())
    monkeypatch.setattr(
        data.urllib.request,
        "urlretrieve",
        lambda url, dest: Path(dest).write_bytes(source.read_bytes()),
    )
    target = env / "downloaded"
    monkeypatch.setenv("CIFAR_DATA_DIR", str(target))
    monkeypatch.setenv("CIFAR_PROJECT_ROOT", str(env))  # manifest resolved under tmp, not the repo
    get_settings.cache_clear()
    assert "X_train.npy" in invoke("download-data").output
    manifest = env / "data" / "manifest.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(json.dumps({f: file_digest(target / f) for f in FILES}))
    assert "OK" in invoke("verify-data").output
    manifest.write_text(json.dumps({f: "0" * 64 for f in FILES}))
    result = runner.invoke(app, ["verify-data"])
    assert result.exit_code == 1
    assert "DIFF" in result.output


def test_export_samples(env: Path):
    invoke("export-samples", "--count", "20", "--out-dir", str(env / "samples"))
    assert len(list((env / "samples").glob("*.png"))) == 20


def test_serve_uses_settings(env: Path, monkeypatch: pytest.MonkeyPatch):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(kw))
    invoke("serve", "--port", "9100")
    assert calls == {"host": "127.0.0.1", "port": 9100, "log_level": "warning"}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"label": 0}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def server_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_benchmark(env: Path, server_url: str):
    invoke("train", "--experiment", "resnet")
    invoke("benchmark", "--url", server_url)
    result = json.loads((env / "reports" / "benchmark.json").read_text())
    assert result["cpu"]["single_image"]["p50_ms"] > 0
    assert result["http"]["requests"] == 200
