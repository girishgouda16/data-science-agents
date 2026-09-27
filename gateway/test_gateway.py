"""Self-check for the gateway — no live orchestrator/LLM (agent_client is
faked), no alembic (the schema is created directly), no Keycloak (a local RSA
key stands in for the realm's JWKS).

Run: python -m pytest -q gateway/test_gateway.py   (or python -m gateway.test_gateway)
"""

import os
import tempfile
import time

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkstemp(suffix='.db')[1]}"
os.environ["JWT_SECRET"] = "test-secret-at-least-32-bytes-long-000000"
os.environ["GATEWAY_POLL_SECONDS"] = "0.05"
os.environ.setdefault("AGENTIC_ML_DATA_DIR", tempfile.mkdtemp(prefix="agentic-ml-gw-"))
os.environ.setdefault(
    "MLFLOW_TRACKING_URI", f"file:{tempfile.mkdtemp(prefix='agentic-ml-gw-mlruns-')}"
)

import json  # noqa: E402

import jwt  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import gateway.app as g  # noqa: E402
from core import auth, runtime  # noqa: E402
from core.auth import issue_token  # noqa: E402
from core.config import settings  # noqa: E402
from core.db import get_engine, metadata, save_a2a_task  # noqa: E402

metadata.create_all(get_engine())  # skip alembic for this smoke test


# ── fake orchestrator ────────────────────────────────────────────────────────
class FakeOrchestrator:
    """start_turn / get_turn / cancel_turn as gateway/agent_client has them.
    `script` is what the next turn does: a list of states it reports, the
    last one final."""

    def __init__(self):
        self.sent, self.cancelled, self.script, self.fail = [], [], ["completed"], False
        self.tasks = {}

    async def start_turn(self, message, task_id, context_id, metadata):
        if self.fail:
            raise ConnectionError("orchestrator down")
        self.sent.append(
            {
                "message": message,
                "task_id": task_id,
                "context_id": context_id,
                "metadata": metadata,
            }
        )
        tid = f"t-{len(self.sent)}"
        self.tasks[tid] = {"states": list(self.script), "text": f"echo: {message}"}
        return self._report(tid)

    def _report(self, tid):
        task = self.tasks[tid]
        state = task["states"].pop(0) if len(task["states"]) > 1 else task["states"][0]
        return {
            "text": "" if state == "working" else task["text"],
            "task_id": tid,
            "context_id": "ctx-1",
            "state": state,
        }

    async def get_turn(self, task_id):
        return self._report(task_id)

    async def cancel_turn(self, task_id):
        self.cancelled.append(task_id)
        self.tasks[task_id]["states"] = ["canceled"]
        return {
            "text": "",
            "task_id": task_id,
            "context_id": "ctx-1",
            "state": "canceled",
        }


orch = FakeOrchestrator()
g.agent_client.start_turn, g.agent_client.get_turn, g.agent_client.cancel_turn = (
    orch.start_turn,
    orch.get_turn,
    orch.cancel_turn,
)

client = TestClient(g.app)
client.__enter__()  # run the lifespan and keep one event loop for background followers
headers = {"Authorization": f"Bearer {issue_token('test-user')}"}
other_user_headers = {"Authorization": f"Bearer {issue_token('someone-else')}"}

SAMPLE_CSV = g.UPLOAD_DIR / "test_gateway_sample.csv"
SAMPLE_CSV.write_text("a,b,y\n1,2,0\n3,4,1\n")


def _wait_until_done(session_id, h=headers, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/sessions/{session_id}", headers=h).json()
        if body["status"] != "working":
            return body
        time.sleep(0.05)
    raise AssertionError(f"session {session_id} still working after {timeout}s")


# ── sessions ─────────────────────────────────────────────────────────────────
def test_rejects_missing_auth():
    assert client.post("/api/v1/sessions", json={"message": "hello"}).status_code == 401


def test_full_session_round_trip():
    orch.script = ["completed"]
    resp = client.post(
        "/api/v1/sessions",
        json={"message": "Classify fraud in transactions.csv"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    session_id = body["session_id"]
    assert (
        body["status"] == "completed"
        and body["reply"] == "echo: Classify fraud in transactions.csv"
    )
    assert body["title"] == "Classify fraud in transactions.csv"

    resp = client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"message": "go"},
        headers=headers,
    )
    assert resp.json()["reply"] == "echo: go"
    assert (
        len(
            client.get(f"/api/v1/sessions/{session_id}", headers=headers).json()[
                "history"
            ]
        )
        == 4
    )


def test_a_long_turn_runs_in_the_background():
    """No request is held open while a model trains: POST returns "working",
    the reply lands when the orchestrator's task finishes."""
    orch.script = ["working", "working", "working", "completed"]
    body = client.post(
        "/api/v1/sessions", json={"message": "train a big model"}, headers=headers
    ).json()
    assert body["status"] == "working" and body["history"][-1]["role"] == "user"
    busy = client.post(
        f"/api/v1/sessions/{body['session_id']}/messages",
        json={"message": "again"},
        headers=headers,
    )
    assert busy.status_code == 409, "one turn at a time per chat"
    done = _wait_until_done(body["session_id"])
    assert done["status"] == "completed" and done["reply"] == "echo: train a big model"


def test_a_question_keeps_the_task_for_the_answer():
    orch.script = ["input-required"]
    body = client.post(
        "/api/v1/sessions", json={"message": "train"}, headers=headers
    ).json()
    assert body["status"] == "input-required"
    orch.script = ["completed"]
    client.post(
        f"/api/v1/sessions/{body['session_id']}/messages",
        json={"message": "churned"},
        headers=headers,
    )
    assert orch.sent[-1]["task_id"] == "t-" + str(len(orch.sent) - 1), (
        "the answer continues the paused task"
    )


def test_cancel_stops_the_running_turn():
    orch.script = ["working"]
    body = client.post(
        "/api/v1/sessions", json={"message": "train forever"}, headers=headers
    ).json()
    assert (
        client.post(
            f"/api/v1/sessions/{body['session_id']}/cancel", headers=other_user_headers
        ).status_code
        == 404
    )
    resp = client.post(f"/api/v1/sessions/{body['session_id']}/cancel", headers=headers)
    assert resp.status_code == 200 and resp.json()["status"] == "canceled"
    assert orch.cancelled[-1] == f"t-{len(orch.sent)}", (
        "the running orchestrator task itself was cancelled"
    )
    assert "Stopped at your request" in resp.json()["reply"]
    assert (
        client.post(
            f"/api/v1/sessions/{body['session_id']}/cancel", headers=headers
        ).status_code
        == 409
    )


def test_every_message_carries_a_signed_user_context_and_the_chat_id():
    orch.script = ["completed"]
    body = client.post(
        "/api/v1/sessions",
        json={"message": "hi", "project": "retention"},
        headers=headers,
    ).json()
    metadata = orch.sent[-1]["metadata"]
    claims = jwt.decode(
        metadata["user_context"],
        os.environ["JWT_SECRET"],
        algorithms=["HS256"],
        audience="agentic-ml-agents",
    )
    assert claims["sub"] == "test-user" and claims["project"] == "retention"
    assert metadata["langfuse_session_id"] == body["session_id"]
    assert "user_id" not in metadata, (
        "a bare user id is never sent — agents trust only the signature"
    )


def test_autopilot_rides_the_signed_context_and_can_be_switched_mid_chat():
    def autopilot_claim():
        return jwt.decode(
            orch.sent[-1]["metadata"]["user_context"],
            os.environ["JWT_SECRET"],
            algorithms=["HS256"],
            audience="agentic-ml-agents",
        ).get("autopilot", False)

    orch.script = ["completed", "completed", "completed"]
    body = client.post(
        "/api/v1/sessions", json={"message": "hi", "autopilot": True}, headers=headers
    ).json()
    assert body["autopilot"] is True and autopilot_claim() is True
    sid = body["session_id"]
    client.post(
        f"/api/v1/sessions/{sid}/messages", json={"message": "go on"}, headers=headers
    )
    assert autopilot_claim() is True, (
        "a message without the field keeps the chat's setting"
    )
    body = client.post(
        f"/api/v1/sessions/{sid}/messages",
        json={"message": "ask me", "autopilot": False},
        headers=headers,
    ).json()
    assert body["autopilot"] is False and autopilot_claim() is False


def test_orchestrator_failure_returns_502_not_500():
    orch.fail = True
    try:
        resp = client.post("/api/v1/sessions", json={"message": "hi"}, headers=headers)
        assert (
            resp.status_code == 502
            and "orchestrator unreachable" in resp.json()["detail"]
        )
    finally:
        orch.fail = False


def test_a_restarted_gateway_follows_turns_that_were_still_running():
    orch.script = ["working", "completed"]
    orch.tasks["t-resume"] = {
        "states": ["working", "completed"],
        "text": "finished while we were down",
    }
    save_a2a_task(
        "s-resume",
        "test-user",
        "working",
        {
            "title": "x",
            "history": [{"role": "user", "text": "x"}],
            "running_task_id": "t-resume",
        },
    )
    with TestClient(g.app):  # a second lifespan = a gateway restart
        deadline = time.time() + 5
        while (
            time.time() < deadline and g.get_a2a_task("s-resume")["state"] == "working"
        ):
            time.sleep(0.05)
    record = g.get_a2a_task("s-resume")
    assert (
        record["state"] == "completed"
        and record["artifacts"]["history"][-1]["text"] == "finished while we were down"
    )


def test_list_sessions_only_returns_own_sessions():
    orch.script = ["completed"]
    a = client.post(
        "/api/v1/sessions", json={"message": "mine one"}, headers=headers
    ).json()
    b = client.post(
        "/api/v1/sessions", json={"message": "mine two"}, headers=headers
    ).json()
    client.post(
        "/api/v1/sessions", json={"message": "not mine"}, headers=other_user_headers
    )
    sessions = client.get("/api/v1/sessions", headers=headers).json()["sessions"]
    assert {a["session_id"], b["session_id"]} <= {s["session_id"] for s in sessions}
    assert all(s["title"] != "not mine" for s in sessions)


def test_cannot_read_continue_or_delete_another_users_session():
    orch.script = ["completed"]
    session_id = client.post(
        "/api/v1/sessions", json={"message": "private"}, headers=headers
    ).json()["session_id"]
    assert (
        client.get(
            f"/api/v1/sessions/{session_id}", headers=other_user_headers
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"message": "hijack"},
            headers=other_user_headers,
        ).status_code
        == 404
    )
    assert (
        client.delete(
            f"/api/v1/sessions/{session_id}", headers=other_user_headers
        ).status_code
        == 404
    )
    assert (
        client.get(f"/api/v1/sessions/{session_id}", headers=headers).status_code == 200
    )
    assert (
        client.delete(f"/api/v1/sessions/{session_id}", headers=headers).status_code
        == 200
    )
    assert (
        client.delete(f"/api/v1/sessions/{session_id}", headers=headers).status_code
        == 404
    )


# ── files ────────────────────────────────────────────────────────────────────
def test_session_with_file_path_prefixes_dataset_line():
    orch.script = ["completed"]
    resp = client.post(
        "/api/v1/sessions",
        json={"message": "analyse this", "file_path": str(SAMPLE_CSV)},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert (
        resp.json()["reply"] == f"echo: Dataset: {SAMPLE_CSV.resolve()}\nanalyse this"
    )
    assert resp.json()["title"] == "analyse this", "title is the user's own words"
    shown = resp.json()["history"][0]
    assert shown == {
        "role": "user",
        "text": "analyse this",
        "attachment": SAMPLE_CSV.name,
    }, "no server path in the chat"


def test_bad_file_paths_rejected():
    for path, status in (
        ("/etc/passwd", 400),
        (str(g.UPLOAD_DIR / ".." / ".." / "etc" / "passwd"), 400),
        (str(g.UPLOAD_DIR / "does-not-exist.csv"), 400),
    ):
        resp = client.post(
            "/api/v1/sessions",
            json={"message": "hi", "file_path": path},
            headers=headers,
        )
        assert resp.status_code == status, (path, resp.status_code)


def test_upload_then_session_round_trip():
    anonymous = client.post(
        "/api/v1/upload", files={"file": ("mine.csv", b"a,b\n1,2\n", "text/csv")}
    )
    assert anonymous.status_code == 401, "uploads need a token like every other route"
    resp = client.post(
        "/api/v1/upload",
        files={"file": ("mine.csv", b"a,b\n1,2\n", "text/csv")},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    uploaded_path = resp.json()["file_path"]
    for name, body in (("model.pkl", b"a,b\n1,2\n"), ("data.csv", b"\x80\x04\x95\0\0")):
        refused = client.post(
            "/api/v1/upload", files={"file": (name, body, "text/csv")}, headers=headers
        )
        assert refused.status_code == 415, (
            f"{name} must be refused: {refused.status_code}"
        )
    assert uploaded_path.startswith(str(g.UPLOAD_DIR.resolve()))
    assert (
        client.post(
            "/api/v1/sessions",
            json={"message": "hi", "file_path": uploaded_path},
            headers=headers,
        ).status_code
        == 200
    )
    stolen = client.post(
        "/api/v1/sessions",
        json={"message": "hi", "file_path": uploaded_path},
        headers=other_user_headers,
    )
    assert stolen.status_code == 403, (
        "a colleague's upload is not usable in your session"
    )
    got = client.get(
        "/api/v1/download", params={"path": uploaded_path}, headers=headers
    )
    assert (
        got.status_code == 200
        and got.content == b"a,b\n1,2\n"
        and 'filename="mine.csv"' in got.headers["content-disposition"]
    )
    for path, h in (
        (uploaded_path, other_user_headers),
        ("/etc/passwd", headers),
        (str(g.UPLOAD_DIR / "test-user" / ".." / ".." / "training_runs.db"), headers),
    ):
        assert (
            client.get("/api/v1/download", params={"path": path}, headers=h).status_code
            == 404
        ), path


# ── identity, runs, config ───────────────────────────────────────────────────
def test_me_and_auth_config():
    assert client.get("/api/v1/auth/config").json() == {"mode": "dev"}
    me = client.get("/api/v1/me", headers=headers).json()
    assert me["user"] == "test-user" and me["admin"] is False
    admin = {"Authorization": f"Bearer {issue_token('boss', roles=['ml-admin'])}"}
    assert client.get("/api/v1/me", headers=admin).json()["admin"] is True


def test_runs_view_shows_only_the_callers_runs():
    runs_dir = g.toolguard.data_dir() / "runs"
    for owner in ("test-user", "someone-else"):
        run = runs_dir / f"run-of-{owner}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "meta.json").write_text(json.dumps({"owner": owner, "target": "y"}))
    owners = {
        r["owner"] for r in client.get("/api/v1/runs", headers=headers).json()["runs"]
    }
    assert owners == {"test-user"}, owners


def _registered(name: str, owner: str, champion: bool = False) -> int:
    import mlflow
    from sklearn.dummy import DummyClassifier

    runtime.bind(owner)
    try:
        mlflow.set_tracking_uri(g.registry.tracking_uri())
        mlflow.set_experiment("housekeeping")
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(DummyClassifier().fit([[0], [1]], [0, 1]), "model")
        version = g.registry.register(run.info.run_id, name)["version"]
        if champion:
            assert "error" not in g.registry.promote(name, version, "champion")
        return version
    finally:
        runtime.bind(None)


def test_housekeeping_deletes_only_what_is_yours_and_not_in_use():
    runs_dir = g.toolguard.data_dir() / "runs"

    def run(run_id, owner, registry=None):
        (runs_dir / run_id).mkdir(parents=True, exist_ok=True)
        (runs_dir / run_id / "meta.json").write_text(
            json.dumps({"owner": owner, "registry": registry})
        )

    run("hk-mine", "test-user")
    run("hk-theirs", "someone-else")
    assert client.delete("/api/v1/runs/hk-mine", headers=headers).status_code == 200
    assert not (runs_dir / "hk-mine").exists()
    assert client.delete("/api/v1/runs/hk-theirs", headers=headers).status_code == 403
    assert client.delete("/api/v1/runs/no-such-run", headers=headers).status_code == 404
    assert all(
        g.runs.delete_run(bad)["status"] == 404 for bad in ("..", "../runs", "a/b")
    ), "never a path"

    version = _registered("hk-model", "test-user")
    run("hk-registered", "test-user", {"name": "hk-model", "version": version})
    assert (
        client.delete("/api/v1/runs/hk-registered", headers=headers).status_code == 409
    ), "a registered run stays"
    _registered("hk-serving", "test-user", champion=True)
    assert (
        client.delete("/api/v1/models/hk-serving", headers=headers).status_code == 409
    ), "champion is serving"
    _registered("hk-theirs-model", "someone-else")
    assert (
        client.delete("/api/v1/models/hk-theirs-model", headers=headers).status_code
        == 403
    )
    assert client.delete("/api/v1/models/hk-model", headers=headers).status_code == 200
    assert (
        client.delete("/api/v1/runs/hk-registered", headers=headers).status_code == 200
    ), "free once its model is gone"


def test_keycloak_tokens_are_verified_against_the_realm():
    """RS256 against the realm's key: right issuer + audience + unexpired, or
    401. The user is preferred_username; realm roles become roles."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer = "http://keycloak.test/realms/ml-agents"

    class FakeJWKS:
        def get_signing_key_from_jwt(self, _token):
            return type("K", (), {"key": key.public_key()})()

    def token(**over):
        claims = {
            "iss": issuer,
            "aud": "ml-agents",
            "sub": "f81d4fae-uuid",
            "preferred_username": "dana",
            "realm_access": {"roles": ["ml-admin", "offline_access"]},
            "exp": int(time.time()) + 300,
            **over,
        }
        return {"Authorization": "Bearer " + jwt.encode(claims, key, algorithm="RS256")}

    saved = settings.oidc_issuer
    settings.oidc_issuer = issuer
    auth._jwks.cache_clear()
    original = auth._jwks
    auth._jwks = lambda: FakeJWKS()
    try:
        me = client.get("/api/v1/me", headers=token()).json()
        assert (
            me["user"] == "dana" and "ml-admin" in me["roles"] and me["admin"] is True
        )
        assert client.get("/api/v1/auth/config").json()["mode"] == "oidc"
        for bad in (
            token(iss="http://evil/realms/x"),
            token(aud="account"),
            token(exp=int(time.time()) - 120),
            headers,
        ):
            assert client.get("/api/v1/me", headers=bad).status_code == 401, (
                "wrong issuer/audience/expired/dev token"
            )
    finally:
        settings.oidc_issuer = saved
        auth._jwks = original


def test_user_context_round_trips_through_an_agent():
    """What the gateway signs is exactly what an agent's bind_user accepts."""
    ctx = runtime.sign_user_context("dana", "retention", ["ml-admin"], autopilot=True)
    user = runtime.bind_user(type("M", (), {"metadata": {"user_context": ctx}})())
    assert (user, runtime.project(), runtime.is_admin(), runtime.autopilot()) == (
        "dana",
        "retention",
        True,
        True,
    )
    assert runtime.mcp_env()[runtime.AUTOPILOT_ENV] == "1", (
        "the tool process sees it too (approval audit tag)"
    )
    runtime.bind(None)
    assert not runtime.autopilot()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q", "-p", "no:warnings"]))
