"""CLI 진단 출력 테스트.

라이브 스모크에서 사용자가 헛다리를 짚지 않으려면 두 상황이 눈에 띄어야 합니다:
설정 오류(FatalError)와 usage 미제공.
"""

from __future__ import annotations

import debate.cli as cli
from debate.provider import FakeBehavior, FakeProvider


def _run(monkeypatch, fake: FakeProvider, argv: list[str]) -> int:
    monkeypatch.setattr(cli, "_build_fake", lambda specs, lat: fake)
    return cli.main(argv)


BASE = ["run", "--provider-override", "fake", "--topic", "T",
        "--agent", "m1", "--agent", "m2"]


def test_fatal_error_prints_config_hint(monkeypatch, capsys):
    """401 은 참가자 단위로 격리되는 바람에 헤드라인이 '참가자 부족'으로 보입니다.
    실제 원인이 키/모델 ID 라는 걸 짚어주지 않으면 엉뚱한 곳을 뒤지게 됩니다."""
    fake = FakeProvider({
        "m1": FakeBehavior(latency_ms=1, fail_always=True, fail_mode="fatal"),
        "m2": FakeBehavior(latency_ms=1, fail_always=True, fail_mode="fatal"),
    })
    code = _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert code == 1
    assert "aborted_insufficient_participants" in out
    assert "FatalError 는 재시도로 해결되지 않는" in out
    assert "API_KEY" in out and "BASE_URL" in out


def test_transient_failure_does_not_print_config_hint(monkeypatch, capsys):
    """500 은 설정 문제가 아니므로 힌트를 띄우면 오히려 오도합니다."""
    fake = FakeProvider({
        "m1": FakeBehavior(latency_ms=1),
        "m2": FakeBehavior(latency_ms=1, fail_always=True, fail_mode="500"),
    })
    _run(monkeypatch, fake, BASE)
    assert "재시도로 해결되지 않는" not in capsys.readouterr().out


def test_zero_usage_is_flagged(monkeypatch, capsys):
    """usage 를 안 돌려주는 엔드포인트는 성공(exit 0)하면서 집계만 0 이 됩니다.
    비용 추적이 조용히 죽는 유일한 경로라 반드시 경고해야 합니다."""
    class NoUsage(FakeProvider):
        async def chat(self, req):
            from dataclasses import replace
            from debate.models import Usage
            return replace(await super().chat(req), usage=Usage(0, 0))

    code = _run(monkeypatch, NoUsage({"m1": FakeBehavior(latency_ms=1),
                                      "m2": FakeBehavior(latency_ms=1)}), BASE)
    out = capsys.readouterr().out

    assert code == 0                      # 실패가 아님 — 그래서 더 놓치기 쉬움
    assert "토큰이 0 입니다" in out


def test_healthy_run_prints_no_hints(monkeypatch, capsys):
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=1), "m2": FakeBehavior(latency_ms=1)})
    code = _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert code == 0
    assert "힌트:" not in out
    assert "failed=0" in out
