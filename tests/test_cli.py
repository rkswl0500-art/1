"""CLI 진단 출력 테스트.

라이브 스모크에서 사용자가 헛다리를 짚지 않으려면 두 상황이 눈에 띄어야 합니다:
설정 오류(FatalError)와 usage 미제공.
"""

from __future__ import annotations

import debate.cli as cli
from debate.provider import FakeBehavior, FakeProvider


def _run(monkeypatch, fake: FakeProvider, argv: list[str]) -> int:
    monkeypatch.setattr(cli, "_build_fake", lambda *a, **k: fake)
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


def test_truncated_utterance_is_flagged_loudly(monkeypatch, capsys):
    """토론에서 잘린 발언은 곧 잘못된 판정입니다. 기록만 하고 안 보여주면
    사용자는 Judge 결과가 왜 이상한지 영원히 모릅니다."""
    class Truncating(FakeProvider):
        async def chat(self, req):
            from dataclasses import replace
            return replace(await super().chat(req), finish_reason="length")

    code = _run(monkeypatch, Truncating({"m1": FakeBehavior(latency_ms=1),
                                         "m2": FakeBehavior(latency_ms=1)}), BASE)
    out = capsys.readouterr().out

    assert code == 0                                  # 실패가 아니라서 더 위험
    assert "⚠ 잘림(finish_reason=length)" in out
    assert "발언이 잘렸습니다" in out
    assert "DEBATE_MAX_OUTPUT_TOKENS" in out


def test_normal_finish_reason_prints_no_truncation_warning(monkeypatch, capsys):
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=1), "m2": FakeBehavior(latency_ms=1)})
    _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert "잘림" not in out
    assert "finish_reason='stop'" in out              # 항상 보이게


def test_progress_shows_fast_agent_while_slow_one_runs(monkeypatch, capsys):
    """지연 편차가 30배(관측: 54.3s vs 1.8s)라 라운드는 가장 느린 참가자에
    묶입니다. 빠른 쪽 완료가 즉시 보여야 멈춘 게 아니라 기다리는 중임을 압니다."""
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=200),
                         "m2": FakeBehavior(latency_ms=5)})
    _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert "[R1] 시작" in out
    assert "참가자 B 완료" in out and "[1/2]" in out    # 빠른 쪽이 먼저
    assert out.index("참가자 B 완료") < out.index("참가자 A 완료")


def test_timeout_warning_silent_on_defaults(monkeypatch, capsys):
    """기본값에서 뜨는 경고는 경고가 아니라 노이즈입니다."""
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=1), "m2": FakeBehavior(latency_ms=1)})
    _run(monkeypatch, fake, BASE)
    assert "라운드 타임아웃" not in capsys.readouterr().out


def test_timeout_warning_fires_on_inconsistent_override(monkeypatch, capsys):
    monkeypatch.setenv("DEBATE_ROUND_TIMEOUT_S", "60")
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=1), "m2": FakeBehavior(latency_ms=1)})
    _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert "라운드 타임아웃 60s < 재시도 예산 360s" in out
