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


def test_models_endpoint_classifies_without_hiding(client):
    """숨기지 않고 분류만 합니다 — 목록에서 빼면 패턴 표에 없는 새 모델을
    못 쓰게 됩니다."""
    j = client.get("/models?provider=fake").json()

    assert set(j) == {"provider", "models", "non_chat"}
    assert j["models"]                      # 전부 그대로 들어 있음
    assert isinstance(j["non_chat"], dict)
    for flagged in j["non_chat"]:
        assert flagged in j["models"]       # 분류된 것도 목록에 남아 있어야 합니다


# ── 다기기 모드(1단계: Tailscale) 설정 ──────────────────────────────────────

def test_bind_address_defaults_to_loopback():
    """기본값이 루프백이어야 합니다. 이 기본값이 흘러서 0.0.0.0 이 되면,
    서버를 띄운 사람이 모르는 채로 외부에 열립니다."""
    s = api.Settings()
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.host_is_loopback


def test_bind_address_comes_from_env(monkeypatch):
    """DEBATE_HOST 로 바꿀 수 있어야 합니다 — 안 그러면 폰에서 못 붙습니다.

    실측(lsof): DEBATE_HOST=127.0.0.1 은 `127.0.0.1:8111 (LISTEN)`,
    DEBATE_HOST=0.0.0.0 은 `*:8112 (LISTEN)`. 인터페이스 주소 192.0.2.2 로
    접속하면 전자는 ConnectionRefusedError, 후자는 연결됩니다.
    """
    monkeypatch.setenv("DEBATE_HOST", "0.0.0.0")
    monkeypatch.setenv("DEBATE_PORT", "9001")
    s = api.Settings()
    assert (s.host, s.port) == ("0.0.0.0", 9001)
    assert not s.host_is_loopback


@pytest.mark.parametrize("host,loopback", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True),
    ("0.0.0.0", False), ("100.101.102.103", False),   # tailscale CGNAT 대역
])
def test_loopback_classification(host, loopback):
    assert api.Settings(host=host).host_is_loopback is loopback


def test_api_gate_default_does_not_stall_non_interactive_callers():
    """API 기본값은 0 — 대기 없음입니다(무한 대기가 아닙니다).

    한 번 300 으로 올렸다가 되돌렸습니다. 웹 UI 는 `gatesec` 를 항상 실어 보내서
    이 기본값이 폰 사용자에게 닿지 않고, 지시를 넣지 않는 스크립트만 라운드
    경계마다 기다리게 됩니다. 사람이 끼어들 시간은 UI 기본값에서 조정합니다.
    """
    body = api.CreateDebate(topic="t", participants=[
        api.Participant(provider="fake", model="fake-a"),
        api.Participant(provider="fake", model="fake-b"),
    ])
    assert body.gate_timeout_s == 0.0


# ── tailscale serve 경유 — 실측 기반 ────────────────────────────────────────

def test_tailnet_client_is_refused_even_though_the_bind_is_loopback(env):
    """`tailscale serve` 를 거치면 클라이언트 주소가 **폰의 tailnet IP** 입니다.

    실측(사용자 환경, 2026-10): `100.75.85.54:0`. 포트가 0 인 것까지 그대로
    옮겼습니다 — 프록시가 소켓 주소를 그렇게 넘깁니다.

    여기서 중요한 건 서버가 `127.0.0.1` 에 바인드돼 있어도 이 요청이 들어온다는
    점입니다. 바인드 주소로 "밖에서 닿는가"를 판정했다면 통과시켰을 요청을,
    **요청 출처**로 판정하기 때문에 막습니다.
    """
    with TestClient(api.app, client=("100.75.85.54", 0)) as c:
        r = c.get("/config/providers")

    assert r.status_code == 403
    assert "100.75.85.54" in r.json()["detail"]      # 어디서 왔는지 밝힙니다


def test_config_ui_off_refuses_the_tailnet_client_without_looking_at_origin(env, monkeypatch):
    """보장은 off 가 합니다 — 출처를 보기 **전에** 걸립니다.

    폰에서 404 가 나온 건 이 경로입니다. 403 이 나온다면 off 가 안 먹은 것이고,
    그때는 두 번째 방어선이 대신 막고 있는 상태입니다.
    """
    monkeypatch.setenv("DEBATE_CONFIG_UI", "off")
    with TestClient(api.app, client=("100.75.85.54", 0)) as c:
        r = c.get("/config/providers")

    assert r.status_code == 404


def test_debate_page_stays_open_to_the_tailnet(env):
    """설정만 막고 토론 화면은 열려야 합니다 — 안 그러면 다기기 자체가 무의미합니다."""
    with TestClient(api.app, client=("100.75.85.54", 0)) as c:
        assert c.get("/").status_code == 200


# ── 슬롯의 사고 수준 전송 정책 ──────────────────────────────────────────────

def test_reasoning_policy_survives_a_save_that_omits_it(client, env):
    """화면은 키처럼 정책도 '비우면 유지' 입니다. 이게 없으면 .env 에 손으로
    적어둔 정책이 화면에서 저장할 때마다 지워집니다."""
    client.put("/config/providers", json={"slots": [
        {"name": "groq", "base_url": "https://x.test/v1", "api_key": REAL_KEY,
         "reasoning": "off"}]})
    client.put("/config/providers", json={"slots": [          # reasoning 없이 저장
        {"name": "groq", "base_url": "https://y.test/v1", "api_key": ""}]})

    assert "DEBATE_PROVIDER_1_REASONING=off" in env.read_text(encoding="utf-8")
    slots = client.get("/config/providers").json()["slots"]
    assert slots[0]["reasoning"] == "off"


def test_reasoning_policy_follows_the_slot_when_renumbered(client, env):
    """저장할 때 슬롯 번호를 1 부터 다시 매깁니다. 정책이 번호에 묶여 있으면
    순서를 바꾸는 순간 **다른 프로바이더**에 붙습니다."""
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://g.test/v1", "api_key": REAL_KEY},
        {"name": "groq", "base_url": "https://q.test/v1", "api_key": REAL_KEY,
         "reasoning": "off"}]})
    client.put("/config/providers", json={"slots": [          # gemini 를 삭제
        {"name": "groq", "base_url": "https://q.test/v1", "api_key": ""}]})

    by_name = {s["name"]: s["reasoning"] for s in
               client.get("/config/providers").json()["slots"]}
    assert by_name == {"groq": "off"}


def test_default_policy_writes_no_line(client, env):
    client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://g.test/v1", "api_key": REAL_KEY}]})
    assert "REASONING" not in env.read_text(encoding="utf-8")


def test_unknown_policy_is_refused(client, env):
    r = client.put("/config/providers", json={"slots": [
        {"name": "gemini", "base_url": "https://g.test/v1", "api_key": REAL_KEY,
         "reasoning": "sometimes"}]})
    assert r.status_code == 400


def test_stray_reasoning_line_outside_the_block_is_removed():
    """구역 밖에 `_1_REASONING` 이 남으면 번호를 다시 매긴 뒤 엉뚱한 슬롯에
    붙습니다. NAME/KEY 와 똑같이 걷어냅니다."""
    existing = "DEBATE_PROVIDER_1_REASONING=off\nOTHER=1\n"
    out = merge(existing, [SlotInput("groq", "https://q.test/v1", "k")])
    assert "DEBATE_PROVIDER_1_REASONING" not in out
    assert "OTHER=1" in out


def test_env_reasoning_is_validated(tmp_path):
    from debate.config import load_provider_slots
    from debate.models import ConfigError

    env = {"DEBATE_PROVIDER_1_NAME": "groq", "DEBATE_PROVIDER_1_BASE_URL": "https://q.test/v1",
           "DEBATE_PROVIDER_1_REASONING": "maybe"}
    with pytest.raises(ConfigError, match="REASONING"):
        load_provider_slots(env, env_file=None)

    env["DEBATE_PROVIDER_1_REASONING"] = "ON"
    assert load_provider_slots(env, env_file=None).get("groq").reasoning == "on"
