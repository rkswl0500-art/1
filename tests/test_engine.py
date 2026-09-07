"""Engine 테스트: 병렬성, 세마포어, 실패 격리.

fake 프로바이더를 쓰는 이유는 편의가 아니라 정확성입니다. "라운드가 병렬인가",
"3회 실패하면 그 참가자만 빠지는가" 같은 항목은 실제 모델로는 재현이 안 돼서
검증이 성립하지 않습니다.
"""

from __future__ import annotations

import time

import pytest

from debate.agent import DEBATE_RULES, Agent, build_header
from debate.config import PricingTable
from debate.cost import CostMeter
from debate.engine import DebateEngine
from debate.models import AgentDropped, AgentSpec, DebateConfig, UtteranceCompleted
from debate.provider import FakeBehavior, FakeProvider, MeteredProvider, RetryingProvider

PRICING = PricingTable({})


def _spec(i: int, model: str) -> AgentSpec:
    return AgentSpec(
        id=f"p{i}", label=f"참가자 {chr(65 + i - 1)}",
        provider="fake", model=model, persona="토론자",
    )


async def _no_sleep(_: float) -> None:
    return None


def _agents(behaviors: dict[str, FakeBehavior], *, attempts: int = 3):
    fake = FakeProvider(behaviors)
    meter = CostMeter(PRICING, "d_test")
    shared = MeteredProvider(
        RetryingProvider(fake, attempts=attempts, sleep=_no_sleep), meter
    )
    specs = [_spec(i, m) for i, m in enumerate(behaviors, start=1)]
    return [Agent(s, shared, PRICING) for s in specs], specs, meter


def _cfg(specs, **kw) -> DebateConfig:
    return DebateConfig(topic="원격근무는 생산성을 높이는가",
                        participants=tuple(specs), rounds=1, **kw)


# ── 병렬성 ───────────────────────────────────────────────────────────────────


async def test_round_runs_in_parallel_not_sequentially():
    agents, specs, _ = _agents({
        "m1": FakeBehavior(latency_ms=200),
        "m2": FakeBehavior(latency_ms=200),
        "m3": FakeBehavior(latency_ms=200),
    })
    started = time.perf_counter()
    result = await DebateEngine(agents).run(_cfg(specs, max_concurrency=3))
    elapsed_ms = (time.perf_counter() - started) * 1000

    rnd = result.rounds[0]
    assert rnd.sum_latency_ms >= 600      # 직렬이었다면 이만큼 걸림
    assert elapsed_ms < 450               # 실제로는 한 번의 대기만큼만
    assert rnd.waves == 1


async def test_semaphore_forces_waves():
    """참가자 5 / 동시성 3 이면 wall 은 max 가 아니라 2 웨이브입니다.

    이걸 모르면 'wall ≈ max(latency)' 라는 합격 기준이 정상 동작을 실패로
    판정합니다.
    """
    agents, specs, _ = _agents({f"m{i}": FakeBehavior(latency_ms=100) for i in range(1, 6)})
    result = await DebateEngine(agents).run(_cfg(specs, max_concurrency=3))
    rnd = result.rounds[0]

    assert rnd.waves == 2
    assert rnd.wall_ms >= 200             # 2 웨이브
    assert rnd.wall_ms < rnd.sum_latency_ms


# ── 실패 격리 (요구사항 7) ───────────────────────────────────────────────────


async def test_one_agent_failing_does_not_stop_the_debate():
    agents, specs, _ = _agents({
        "ok1": FakeBehavior(latency_ms=10),
        "boom": FakeBehavior(latency_ms=10, fail_always=True),
        "ok2": FakeBehavior(latency_ms=10),
    })
    events = []
    result = await DebateEngine(agents, sink=lambda e: _collect(events, e)).run(
        _cfg(specs, max_concurrency=3)
    )

    assert result.status == "completed"
    assert result.dropped == ("p2",)
    statuses = {u.agent_id: u.status for u in result.rounds[0].utterances}
    assert statuses == {"p1": "ok", "p2": "failed", "p3": "ok"}
    assert [e.agent_id for e in events if isinstance(e, AgentDropped)] == ["p2"]
    # 살아남은 참가자의 발언은 정상적으로 방출됨
    assert len([e for e in events if isinstance(e, UtteranceCompleted)]) == 2


async def test_agent_recovering_within_retry_budget_is_not_dropped():
    agents, specs, meter = _agents({
        "ok": FakeBehavior(latency_ms=10),
        "flaky": FakeBehavior(latency_ms=10, fail_first_n=2),  # 3번째에 성공
    })
    result = await DebateEngine(agents).run(_cfg(specs))
    assert result.dropped == ()
    assert all(u.status == "ok" for u in result.rounds[0].utterances)
    assert max(r.attempts for r in meter.records) == 3


async def test_debate_aborts_when_fewer_than_two_survive():
    agents, specs, _ = _agents({
        "ok": FakeBehavior(latency_ms=10),
        "boom": FakeBehavior(latency_ms=10, fail_always=True),
    })
    result = await DebateEngine(agents).run(_cfg(specs))
    assert result.status == "aborted_insufficient_participants"
    # 중단돼도 진행된 부분은 남습니다
    assert len(result.rounds) == 1


async def test_timeout_drops_only_the_hung_agent():
    agents, specs, _ = _agents({
        "fast": FakeBehavior(latency_ms=10),
        "hung": FakeBehavior(latency_ms=5000),
        "fast2": FakeBehavior(latency_ms=10),
    })
    result = await DebateEngine(agents).run(
        _cfg(specs, max_concurrency=3, round_timeout_s=0.3)
    )
    assert result.dropped == ("p2",)
    assert result.status == "completed"


# ── 범위/불변식 ──────────────────────────────────────────────────────────────


async def test_multi_round_is_refused_rather_than_silently_wrong():
    agents, specs, _ = _agents({"a": FakeBehavior(latency_ms=1), "b": FakeBehavior(latency_ms=1)})
    with pytest.raises(NotImplementedError, match="슬라이스 2"):
        await DebateEngine(agents).run(
            DebateConfig(topic="t", participants=tuple(specs), rounds=3)
        )


async def test_round1_context_carries_no_other_participant_content():
    """라운드 내 블라인드. 슬라이스 1 에서는 R1 이라 자명하지만, 이 불변식이
    깨지는 순간을 슬라이스 2 이전에 잡아두려고 지금부터 검사합니다."""
    agents, specs, _ = _agents({"a": FakeBehavior(latency_ms=1), "b": FakeBehavior(latency_ms=1)})
    engine = DebateEngine(agents)
    cfg = _cfg(specs)
    pack = engine._context_for(agents[0], 1, cfg)

    assert pack.last_round == ()
    assert pack.prior_digest == ""
    assert pack.issues == ()
    rendered = "\n".join(m.content for m in pack.render())
    assert agents[1].label not in rendered


def test_header_uses_anon_label_and_never_the_model_id():
    """익명화 1차 방어선.

    헤더에 모델 ID 가 들어가면 참가자가 자기 정체를 알고 그걸 발언에 흘리며,
    그 발언이 그대로 Judge 프롬프트로 갑니다. 라벨만 들어가야 합니다.
    """
    spec = AgentSpec(id="p1", label="참가자 A", provider="openai",
                     model="gpt-4o-mini-2024-07-18", persona="토론자")
    header = build_header(spec, "원격근무는 생산성을 높이는가")

    assert "참가자 A" in header
    assert spec.model not in header
    assert "openai" not in header.lower()
    # 자기소개 금지 지시가 실제로 전달되는지
    assert "어떤 모델인지" in DEBATE_RULES
    assert "어느 회사가 만들었는지" in DEBATE_RULES


async def _collect(bucket: list, event) -> None:
    bucket.append(event)
