"""회귀 테스트: .env 가 실제로 읽히는가.

이 버그는 슬라이스 1 검증을 그대로 통과했습니다. fake 모드가 프로바이더 슬롯을
쓰지 않아서 라이브 경로가 한 번도 실행되지 않았기 때문입니다. 그래서 여기서는
**os.environ 을 비운 상태**로 파일만 놓고 검증합니다 — 환경변수가 남아 있으면
파일을 안 읽어도 통과해버려서 테스트가 의미를 잃습니다.
"""

from __future__ import annotations

import os

import pytest

from debate.config import ProviderRegistry, load_provider_slots

ENV_BODY = (
    "DEBATE_PROVIDER_1_NAME=gemini\n"
    "DEBATE_PROVIDER_1_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai\n"
    "DEBATE_PROVIDER_1_API_KEY=key-from-file\n"
)


@pytest.fixture(autouse=True)
def _clear_slot_env(monkeypatch):
    """os.environ 에 남은 슬롯 변수를 전부 제거. 이게 없으면 테스트가 무의미합니다."""
    for key in list(os.environ):
        if key.startswith("DEBATE_"):
            monkeypatch.delenv(key, raising=False)


def _write(path, body: str, *, bom: bool) -> None:
    raw = body.encode("utf-8")
    path.write_bytes(b"\xef\xbb\xbf" + raw if bom else raw)


@pytest.mark.parametrize("bom", [False, True], ids=["no-bom", "with-bom"])
def test_env_file_is_actually_read(tmp_path, monkeypatch, bom):
    """원 버그: load_provider_slots 가 os.environ 만 봐서 .env 를 무시했습니다.

    BOM 케이스는 PowerShell 의 `Out-File -Encoding utf8` 이 기본으로 BOM 을
    붙이기 때문입니다. BOM 이 있으면 첫 줄 키가 '\\ufeffDEBATE_PROVIDER_1_NAME'
    이 되어 조용히 인식되지 않습니다.
    """
    _write(tmp_path / ".env", ENV_BODY, bom=bom)
    monkeypatch.chdir(tmp_path)

    # 인자 없이 호출 = CLI 가 실제로 쓰는 경로
    registry = load_provider_slots()

    assert registry.names() == ("gemini",)
    slot = registry.get("gemini")
    assert slot.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert slot.has_key
    assert registry.env_file_found


def test_process_env_overrides_env_file(tmp_path, monkeypatch):
    _write(tmp_path / ".env", ENV_BODY, bom=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEBATE_PROVIDER_1_BASE_URL", "https://override.test/v1")

    assert load_provider_slots().get("gemini").base_url == "https://override.test/v1"


def test_env_file_only_slot_needs_no_process_env(tmp_path, monkeypatch):
    """파일에만 있고 환경변수엔 없는 슬롯이 잡혀야 합니다 — 버그의 핵심."""
    _write(tmp_path / ".env", ENV_BODY, bom=False)
    monkeypatch.chdir(tmp_path)

    assert not any(k.startswith("DEBATE_PROVIDER") for k in os.environ)
    assert len(load_provider_slots()) == 1


def test_missing_env_file_is_not_an_error(tmp_path, monkeypatch):
    """.env 가 없어도 죽지 않습니다. fake 모드는 파일 없이 돌아야 합니다."""
    monkeypatch.chdir(tmp_path)
    registry = load_provider_slots()

    assert registry.names() == ()
    assert not registry.env_file_found


def test_explicit_env_file_none_skips_the_file(tmp_path, monkeypatch):
    _write(tmp_path / ".env", ENV_BODY, bom=False)
    monkeypatch.chdir(tmp_path)

    registry = load_provider_slots(env_file=None)
    assert registry.names() == ()
    assert registry.env_file is None


# ── 진단 메시지 ──────────────────────────────────────────────────────────────


def test_hint_names_the_path_it_looked_for(tmp_path, monkeypatch):
    """'사용 가능: <없음>' 만으로는 원인을 못 짚습니다. 어느 경로를 봤는지 말해야
    파일이 없는 건지, 오타인지, 다른 디렉터리에서 실행한 건지 구분됩니다."""
    monkeypatch.chdir(tmp_path)
    hint = load_provider_slots().source_hint()

    assert str(tmp_path / ".env") in hint
    assert "이 경로에 파일이 없습니다" in hint
    assert "현재 디렉터리" in hint


def test_hint_reports_slots_when_file_was_read(tmp_path, monkeypatch):
    _write(tmp_path / ".env", ENV_BODY, bom=False)
    monkeypatch.chdir(tmp_path)
    hint = load_provider_slots().source_hint()

    assert "슬롯 1개 인식" in hint
    assert "gemini" in hint
    assert "이 경로에 파일이 없습니다" not in hint


def test_build_pool_error_carries_the_hint(tmp_path, monkeypatch):
    """CLI 가 실제로 보여주는 에러에 경로가 들어가야 합니다."""
    from debate.config import PricingTable, Settings
    from debate.cost import CostMeter
    from debate.models import AgentSpec, ConfigError
    from debate.provider import build_pool

    monkeypatch.chdir(tmp_path)
    specs = [AgentSpec(id="p1", label="참가자 A", provider="gemini",
                       model="m", persona="토론자")]

    with pytest.raises(ConfigError) as excinfo:
        build_pool(specs, load_provider_slots(), CostMeter(PricingTable({}), "d"),
                   Settings(_env_file=None))

    message = str(excinfo.value)
    assert "gemini" in message
    assert str(tmp_path / ".env") in message


def test_api_key_from_file_is_not_exposed_in_hint(tmp_path, monkeypatch):
    _write(tmp_path / ".env", ENV_BODY, bom=False)
    monkeypatch.chdir(tmp_path)
    registry = load_provider_slots()

    assert "key-from-file" not in registry.source_hint()
    assert "key-from-file" not in repr(registry.get("gemini"))
    assert isinstance(registry, ProviderRegistry)
