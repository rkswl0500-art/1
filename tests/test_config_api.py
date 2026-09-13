"""프로바이더 설정 UI.

.env 를 쓰고 API 키를 다루는 엔드포인트라, 여기 테스트의 대부분은 기능이 아니라
**새어 나가지 않는지**와 **외부에서 못 건드리는지**입니다.
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from debate import api
from debate.envfile import BEGIN, SlotInput, mask_key, merge, read_raw, save_slots

REAL_KEY = "AQ.Ab8RN6Jz-secret-value-here-xQ2"


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setenv("DEBATE_ENV_PATH", str(path))
    monkeypatch.setitem(api.Settings.model_config, "env_file", str(tmp_path / "none"))
    return path


@pytest.fixture
def client(env):
    # TestClient 기본 클라이언트 주소는 "testclient" 라 가드에 걸립니다.
    # 프로덕션 코드가 그 문자열을 받아주게 하면 안 되므로 테스트에서 맞춥니다.
    with TestClient(api.app, client=("127.0.0.1", 50000)) as c:
        yield c


# ── 키가 브라우저로 나가지 않는가 ───────────────────────────────────────────


def test_stored_key_is_never_returned_to_the_browser(client, env):
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": REAL_KEY}]})

    body = client.get("/config/providers").text

    assert REAL_KEY not in body
    assert "AQ.Ab****...****xQ2" in body      # 마스킹만
    assert env.read_text(encoding="utf-8").count(REAL_KEY) == 1   # 파일에는 있음


def test_blank_key_keeps_the_stored_one(client, env):
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": REAL_KEY}]})
    # base_url 만 바꾸고 키는 비워서 저장
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://y.test/v1", "api_key": ""}]})

    text = env.read_text(encoding="utf-8")
    assert REAL_KEY in text
    assert "https://y.test/v1" in text


def test_a_new_key_replaces_the_old_one(client, env):
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": REAL_KEY}]})
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": "NEW.key.value.9876"}]})

    text = env.read_text(encoding="utf-8")
    assert REAL_KEY not in text
    assert "NEW.key.value.9876" in text


def test_connection_test_response_carries_no_key(client, env):
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": REAL_KEY}]})

    r = client.post("/config/providers/test",
                    json={"name": "gemini", "base_url": "https://127.0.0.1:1/v1"})

    assert REAL_KEY not in r.text            # 실패 메시지에도 섞이면 안 됩니다
    assert r.json()["ok"] is False


# ── 외부에서 못 건드리는가 ──────────────────────────────────────────────────


@pytest.mark.parametrize("header", [
    "x-forwarded-for", "x-real-ip", "forwarded", "x-forwarded-host"])
def test_proxied_requests_are_refused(client, header):
    """설정 편집은 직접 로컬 접속만 받습니다. 프록시를 거쳤다는 표시가 있으면
    거부합니다 — 그 경로로 열어두면 외부 노출 여부를 알 수 없습니다."""
    r = client.get("/config/providers", headers={header: "127.0.0.1"})

    assert r.status_code == 403
    assert "프록시" in r.json()["detail"]


def test_non_loopback_client_is_refused(env, monkeypatch):
    """바인드 주소로는 판별할 수 없어서(scope['server'] 는 연결의 로컬 주소)
    '이 요청이 외부에서 왔는가'를 막습니다."""
    with TestClient(api.app, client=("203.0.113.9", 5555)) as c:
        checks = [
            c.get("/config/providers"),
            c.put("/config/providers", json={"slots": []}),
            c.post("/config/providers/test", json={"base_url": "https://x/v1"}),
        ]
        for r in checks:
            assert r.status_code == 403
            assert "로컬에서만" in r.json()["detail"]


def test_config_ui_can_be_turned_off(env, monkeypatch):
    """여러 기기에서 붙는 구성으로 옮길 때 완전히 닫을 수 있어야 합니다."""
    monkeypatch.setenv("DEBATE_CONFIG_UI", "off")
    with TestClient(api.app, client=("127.0.0.1", 50000)) as c:
        assert c.get("/config/providers").status_code == 404


def test_debate_endpoints_stay_open_to_non_loopback(env):
    """가드는 설정 엔드포인트에만 걸립니다. 토론 조회까지 막을 이유는 없습니다."""
    with TestClient(api.app, client=("203.0.113.9", 5555)) as c:
        assert c.get("/debates/nope").status_code == 404   # 403 이 아니라 404


# ── 검증 ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("slots,fragment", [
    ([{"name": "fake", "base_url": "https://x/v1"}], "예약어"),
    ([{"name": "A B", "base_url": "https://x/v1"}], "소문자"),
    ([{"name": "g", "base_url": "ftp://x"}], "http"),
    ([{"name": "g", "base_url": "https://x/v1"},
      {"name": "g", "base_url": "https://y/v1"}], "중복"),
])
def test_bad_slots_are_rejected(client, slots, fragment):
    r = client.put("/config/providers", json={"slots": slots})
    assert r.status_code == 400
    assert fragment in r.json()["detail"]


def test_rejected_save_does_not_touch_the_file(client, env):
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": REAL_KEY}]})
    before = env.read_bytes()

    client.put("/config/providers", json={"slots": [{"name": "fake", "base_url": "https://x/v1"}]})

    assert env.read_bytes() == before


# ── .env 쓰기 ───────────────────────────────────────────────────────────────


def test_written_env_has_no_bom_and_lf_endings(client, env):
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": REAL_KEY}]})
    raw = env.read_bytes()

    assert not raw.startswith(b"\xef\xbb\xbf")    # PowerShell 이 붙이던 BOM
    assert b"\r\n" not in raw
    assert oct(env.stat().st_mode)[-3:] == "600"


def test_existing_settings_and_comments_survive(client, env):
    env.write_bytes("﻿# 내가 적은 주석\nDEBATE_JUDGE_TIMEOUT_S=300\n".encode("utf-8"))

    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": REAL_KEY}]})
    text = env.read_text(encoding="utf-8")

    assert "# 내가 적은 주석" in text
    assert "DEBATE_JUDGE_TIMEOUT_S=300" in text


def test_repeated_saves_do_not_accumulate_blocks(client, env):
    for _ in range(3):
        client.put("/config/providers", json={"slots": [
            {"name": "gemini", "base_url": "https://x.test/v1", "api_key": REAL_KEY}]})

    text = env.read_text(encoding="utf-8")
    assert text.count(BEGIN) == 1
    assert text.count("DEBATE_PROVIDER_1_NAME") == 1


def test_hand_written_slot_lines_outside_the_block_are_replaced(env):
    env.write_text("DEBATE_PROVIDER_1_NAME=old\nDEBATE_PROVIDER_1_BASE_URL=http://old/v1\n",
                   encoding="utf-8")
    save_slots(env, [SlotInput("new", "https://new/v1", "k")])
    text = env.read_text(encoding="utf-8")

    assert "old" not in text
    assert text.count("DEBATE_PROVIDER_1_NAME") == 1


def test_removing_a_slot_removes_it_from_the_file(client, env):
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": REAL_KEY},
        {"name": "groq", "base_url": "https://g.test/v1", "api_key": "gk-123456789012"}]})
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://x.test/v1", "api_key": ""}]})

    text = env.read_text(encoding="utf-8")
    assert "groq" not in text
    assert "gk-123456789012" not in text
    assert REAL_KEY in text


@pytest.mark.parametrize("key,expected", [
    ("AQ.Ab8RN6Jz-secret-value-here-xQ2", "AQ.Ab****...****xQ2"),
    ("short", "********"),
    ("", ""),
    (None, ""),
])
def test_masking(key, expected):
    assert mask_key(key) == expected


def test_presets_cover_the_providers_we_documented(client):  # noqa: D103
    names = {p["name"] for p in client.get("/config/providers").json()["presets"]}
    assert names == {"gemini", "groq", "openrouter"}
