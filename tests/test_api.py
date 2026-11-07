import base64
import json

from conftest import HR, IT, make_tenant


def ingest(client, h, name, text):
    r = client.post("/v1/documents", json={"name": name, "content": text}, headers=h)
    assert r.status_code == 201, r.text
    return r.json()


def test_auth_required(client):
    assert client.post("/v1/query", json={"question": "hello there"}).status_code == 401
    assert client.get("/v1/documents", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/v1/admin/tenants", json={"tenant_id": "x1", "name": "x"}).status_code == 403


def test_end_to_end_answer_with_citation(client):
    h = make_tenant(client)
    ingest(client, h, "handbook.md", HR)
    ingest(client, h, "it.md", IT)
    r = client.post("/v1/query", json={"question": "How many days of annual leave do employees get?"}, headers=h).json()
    assert "25 days" in r["answer"] and "[" in r["answer"]
    assert r["grounded"] is True and r["sources"][0]["document"] == "handbook.md"


def test_unsupported_question_refuses(client):
    h = make_tenant(client)
    ingest(client, h, "handbook.md", HR)
    r = client.post("/v1/query", json={"question": "What is the airspeed of a laden swallow?"}, headers=h).json()
    assert r["grounded"] is False and "don't know" in r["answer"] and "no_support" in r["flags"]


def test_tenant_isolation(client):
    a, b = make_tenant(client, "acme"), make_tenant(client, "globex")
    ingest(client, a, "handbook.md", HR)
    assert client.get("/v1/documents", headers=b).json()["documents"] == []
    r = client.post("/v1/query", json={"question": "annual leave days"}, headers=b).json()
    assert r["sources"] == [] and r["grounded"] is False
    doc_id = client.get("/v1/documents", headers=a).json()["documents"][0]["id"]
    assert client.delete(f"/v1/documents/{doc_id}", headers=b).status_code == 404  # cannot delete across tenants
    assert client.delete(f"/v1/documents/{doc_id}", headers=a).status_code == 204


def test_reingest_idempotent_and_replace(client):
    h = make_tenant(client)
    assert ingest(client, h, "h.md", HR)["status"] == "created"
    assert ingest(client, h, "h.md", HR)["status"] == "unchanged"
    assert ingest(client, h, "h.md", HR + "\n## Extra\nNew policy about remote work days.")["status"] == "replaced"
    assert len(client.get("/v1/documents", headers=h).json()["documents"]) == 1


def test_prompt_injection_in_documents_is_neutralised(client):
    h = make_tenant(client)
    ingest(client, h, "it.md", IT)
    r = client.post("/v1/query", json={"question": "reveal the system prompt injection test"}, headers=h).json()
    assert any(f.startswith("injection_removed") for f in r["flags"]) or "don't know" in r["answer"]
    assert "Ignore all previous" not in r["answer"]


def test_pii_redacted_in_answer(client):
    h = make_tenant(client)
    ingest(client, h, "handbook.md", HR)
    r = client.post("/v1/query", json={"question": "who do I contact for expense claims help"}, headers=h).json()
    assert "finance@acme.example" not in r["answer"]


def test_streaming_sse(client):
    h = make_tenant(client)
    ingest(client, h, "handbook.md", HR)
    r = client.post("/v1/query/stream", json={"question": "annual leave days per year"}, headers=h)
    events = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
    assert events[-1]["type"] == "done" and any(e["type"] == "token" for e in events)


def test_b64_upload_validation_and_metrics(client):
    h = make_tenant(client)
    b64 = base64.b64encode(HR.encode()).decode()
    assert client.post("/v1/documents", json={"name": "a.md", "content_b64": b64}, headers=h).status_code == 201
    assert client.post("/v1/documents", json={"name": "b.md", "content_b64": "!!!"}, headers=h).status_code == 422
    assert client.post("/v1/documents", json={"name": "c.md"}, headers=h).status_code == 422
    assert client.post("/v1/documents", json={"name": "d.txt", "content": "   "}, headers=h).status_code == 422
    m = client.get("/metrics").text
    assert "http_requests_total" in m and "documents_ingested_total" in m


def test_rate_limit():
    from fastapi.testclient import TestClient

    from ragplatform.api.app import create_app
    from ragplatform.config import Settings

    c = TestClient(create_app(Settings(db_path=":memory:", admin_key="test-admin-key", rate_limit_per_min=2)))
    h = make_tenant(c)
    assert c.get("/v1/documents", headers=h).status_code == 200
    assert c.get("/v1/documents", headers=h).status_code == 200
    r = c.get("/v1/documents", headers=h)
    assert r.status_code == 429 and "retry-after" in r.headers
