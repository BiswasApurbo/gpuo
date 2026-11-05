import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from gpuo.server import create_app


def test_dashboard_endpoints(home):
    client = TestClient(create_app(interval=0, home=home))
    assert client.get("/").status_code == 200
    o = client.get("/api/overview").json()
    assert o["targets"] == [] and o["jobs"] == []
    assert client.get("/api/jobs/nope").status_code == 404
