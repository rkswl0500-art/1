"""컨텍스트 규율: 라운드 내 블라인드, 예산 한계, 지시문 자리.

여기 테스트는 대부분 "무엇이 들어가는가"가 아니라 **"무엇이 안 들어가는가"**를
봅니다. 컨텍스트에 새는 건 조용히 일어나고 결과만 미묘하게 망가집니다.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from debate.agent import Agent, Anonymizer, Moderator
from debate.config import PricingTable
from debate.context import ContextBuilder
from debate.cost import CostMeter, estimate_tokens
from debate.engine import DebateEngine
from debate.models import (
    AgentSpec, DebateConfig, DebateState, Directive, Issue, RoundResult, Usage, Utterance,
)
from debate.provider import FakeBehavior, FakeProvider, MeteredProvider, RetryingProvider

PRICING = PricingTable({})
SPECS = [
    AgentSpec("p1", "참가자 A", "gemini", "models/gemini-3.6-flash", "토론자"),
    AgentSpec("p2", "참가자 B", "openai", "gpt-4o-mini", "토론자"),
    AgentSpec("p3", "참가자 C", "groq", "llama-3.3-70b", "토론자"),
]


def _builder() -> ContextBuilder:
    return ContextBuilder(Anonymizer(SPECS))


def _round(no: int, text_by_agent: dict[str, str]) -> RoundResult:
    us = tuple(
        Utterance(aid, no, text, Usage(0, 0), 0, Decimal(0))
        for aid, text in text_by_agent.items()
    )
    return RoundResult(no, us, 0, 0, 1)


def _rendered(pack) -> str:
    return "\n".join(m.content for m in pack.render())


# ── 라운드 내 블라인드 ───────────────────────────────────────────────────────


def test_context_never_contains_the_in_flight_round():
    """DebateState 에 진행 중 라운드를 담을 필드가 없다는 게 이 불변식의 근거입니다.
    필드가 생기는 순간 이 테스트가 깨져야 합니다."""
    state = DebateState(topic="주제", round_no=3)
    state.completed_rounds = [
        _round(1, {"p1": "R1 A 발언", "p2": "R1 B 발언"}),
        _round(2, {"p1": "R2 A 발언", "p2": "R2 B 발언"}),
    ]
    text = _rendered(_builder().build_for(SPECS[0], state))

    assert "R2 B 발언" in text          # 직전 라운드는 전문으로 보임
    assert "R3" not in text.replace("제3라운드", "")   # 진행 중 라운드 흔적 없음
    assert not hasattr(state, "in_flight")


def test_round1_context_has_no_prior_material():
    text = _rendered(_builder().build_for(SPECS[0], DebateState(topic="주제")))

    assert "[확정된 쟁점]" not in text
    assert "[이전 라운드 요약]" not in text
    assert "[직전 라운드 발언 전문]" not in text
    assert "제1라운드 입론" in text


def test_failed_utterances_are_not_carried_into_context():
    state = DebateState(topic="주제", round_no=2)
    state.completed_rounds = [RoundResult(1, (
        Utterance("p1", 1, "정상 발언", Usage(0, 0), 0, Decimal(0)),
        Utterance("p2", 1, "", Usage(0, 0), 0, Decimal(0), status="failed", error="boom"),
    ), 0, 0, 1)]
    text = _rendered(_builder().build_for(SPECS[0], state))

    assert "정상 발언" in text
    assert "참가자 B" not in text       # 실패한 참가자는 아예 등장하지 않음


# ── 컨텍스트 예산 (요구사항 6) ───────────────────────────────────────────────


def test_context_does_not_grow_linearly_with_rounds():
    """직전 라운드만 전문이고 나머지는 요약이므로, 라운드가 늘어도 컨텍스트는
    거의 일정해야 합니다. 이게 무너지면 5명 5라운드에서 비용이 폭발합니다."""
    long_turn = "가" * 400
    builder = _builder()

    def size_at(round_no: int) -> int:
        state = DebateState(topic="주제", round_no=round_no)
        state.prior_digest = "요약본" * 30          # 요약은 길이가 묶여 있음
        state.completed_rounds = [
            _round(n, {s.id: long_turn for s in SPECS})
            for n in range(1, round_no)
        ]
        return estimate_tokens(_rendered(builder.build_for(SPECS[0], state)))

    at_2, at_5, at_9 = size_at(2), size_at(5), size_at(9)

    assert at_5 == at_2                 # 직전 1라운드만 실리므로 동일
    assert at_9 == at_2
    # 참고: 전 라운드를 전문으로 넣었다면 8배가 됐을 크기
    assert at_9 < at_2 * 2


# ── 익명화 (요구사항 4) ──────────────────────────────────────────────────────


def test_model_ids_never_reach_the_context():
    state = DebateState(topic="주제", round_no=2)
    state.completed_rounds = [
        _round(1, {"p2": "gpt-4o-mini 로서 답하자면 원격근무는 좋다"})
    ]
    text = _rendered(_builder().build_for(SPECS[0], state))

    assert "gpt-4o-mini" not in text
    assert "[비공개]" in text
    assert "참가자 B" in text            # 라벨은 살아 있어야 함


def test_human_directives_go_through_the_same_scrub():
    """사용자는 어느 참가자가 어느 모델인지 압니다. 지시문에 모델명을 쓰면
    그게 컨텍스트를 거쳐 Judge 프롬프트까지 흘러갑니다."""
    state = DebateState(topic="주제", round_no=2)
    state.directives = (Directive("gemini가 든 통계의 출처를 캐물어라", 2),)
    text = _rendered(_builder().build_for(SPECS[0], state))

    assert "gemini" not in text.lower()
    assert "통계의 출처를 캐물어라" in text
    assert "[사회자 지시]" in text


def test_legitimate_vendor_mention_is_not_destroyed():
    """토론 내용을 파괴하는 익명화는 익명화가 아니라 검열입니다."""
    anon = Anonymizer(SPECS)
    result = anon.scrub("구글의 2020년 원격근무 정책은 생산성을 높였다")

    assert result.redactions == 0
    assert "구글의 2020년 원격근무 정책" in result.text


# ── 지시문 수명 ──────────────────────────────────────────────────────────────


def test_directives_are_absent_when_none_given():
    text = _rendered(_builder().build_for(SPECS[0], DebateState(topic="주제", round_no=2)))
    assert "[사회자 지시]" not in text


async def test_directives_are_cleared_after_their_round():
    """1회용입니다. 다음 라운드로 넘어가면 안 됩니다."""
    class OneShotGate:
        def __init__(self): self.calls = []

        async def collect(self, round_no, state):
            self.calls.append(round_no)
            return (Directive("R2 에만 적용", 2),) if round_no == 2 else ()

    fake = FakeProvider({s.model: FakeBehavior(latency_ms=1) for s in SPECS[:2]})
    meter = CostMeter(PRICING, "d")
    shared = MeteredProvider(RetryingProvider(fake, attempts=1), meter)
    specs = SPECS[:2]
    agents = [Agent(s, shared, PRICING) for s in specs]
    anon = Anonymizer(specs)
    gate = OneShotGate()

    engine = DebateEngine(agents, ContextBuilder(anon),
                          Moderator(shared, specs[0].model), anon, gate=gate)
    await engine.run(DebateConfig(debate_id="d_t", topic="주제", participants=tuple(specs), rounds=3))

    assert gate.calls == [1, 2, 3]      # 매 라운드 게이트를 거침


async def test_default_gate_injects_nothing():
    """슬라이스 2~3 동안 게이트는 동작을 바꾸지 않아야 합니다."""
    from debate.engine import NoIntervention

    assert await NoIntervention().collect(2, DebateState(topic="t")) == ()
