"""저장과 사전 견적."""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal

import pytest

from debate.config import ModelPrice, PricingTable
from debate.cost import CostMeter, Estimator
from debate.judge import Verdict, family_note
from debate.models import (
    AgentSpec, DebateResult, Issue, RoundResult, Usage, Utterance,
)
from debate.storage import SqliteStore

SPECS = (
    AgentSpec("p1", "참가자 A", "gemini", "models/gemini-3.6-flash", "실증주의자"),
    AgentSpec("p2", "참가자 B", "openai", "gpt-4o-mini", "현장주의자"),
)


def _result() -> DebateResult:
    us = (
        Utterance("p1", 1, "A 의 발언", Usage(100, 50), 120, Decimal("0.001"),
                  finish_reason="stop"),
        Utterance("p2", 1, "", Usage(0, 0), 0, Decimal(0), status="failed", error="boom"),
    )
    return DebateResult(
        debate_id="d_test", topic="원격근무", participants=SPECS,
        rounds=(RoundResult(1, us, 130, 120, 1),),
        dropped=("p2",), issues=(Issue("i1", "측정 지표"),),
        warnings=("요약 실패",),
    )


def _meter() -> CostMeter:
    m = CostMeter(PricingTable({}), "d_test")
    m.record(purpose="debate", agent_id="p1", provider="gemini",
             model="models/gemini-3.6-flash", usage=Usage(100, 50),
             latency_ms=120, round_no=1)
    m.record(purpose="judge", agent_id=None, provider="openai", model="gpt-4o-mini",
             usage=Usage(400, 200), latency_ms=900)
    return m


def _verdict() -> Verdict:
    return Verdict(
        per_issue=(), rubric={"참가자 A": {"근거": 7, "논리": 7, "반박": 7, "명료성": 7}},
        winner="참가자 A", margin="narrow", conclusion="A 가 앞섰다",
        dissent="온보딩 논점 미해결", judge_model="gpt-4o-mini",
        prompt_tokens=400, finish_reason="stop",
        prompt_text="참가자 A:\n발언\n\n참가자 B:\n발언",
    )


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "sub" / "debates.db")   # 디렉터리 자동 생성 확인
    yield s
    s.close()


def test_full_record_round_trips(store):
    store.save(_result(), _meter(), _verdict(),
               family_note("gpt-4o-mini", [s.model for s in SPECS]))
    c = sqlite3.connect(store.path); c.row_factory = sqlite3.Row

    d = c.execute("select * from debates").fetchone()
    assert d["topic"] == "원격근무" and d["rounds"] == 1
    assert json.loads(d["warnings"]) == ["요약 실패"]

    assert c.execute("select count(*) from participants").fetchone()[0] == 2
    assert c.execute("select count(*) from utterances").fetchone()[0] == 2
    assert c.execute("select count(*) from llm_calls").fetchone()[0] == 2
    assert c.execute("select count(*) from issues").fetchone()[0] == 1


def test_label_to_model_mapping_is_stored(store):
    """점수를 모델로 되돌리려면 매핑이 있어야 합니다 — DB 에만, 프롬프트에는 절대."""
    store.save(_result(), _meter(), _verdict(), None)
    c = sqlite3.connect(store.path)
    rows = dict(c.execute("select anon_label, model from participants"))

    assert rows == {"참가자 A": "models/gemini-3.6-flash", "참가자 B": "gpt-4o-mini"}


def test_stored_judge_prompt_contains_no_model_names(store):
    """익명화 주장은 실제로 보낸 문자열을 grep 할 수 있어야 검증됩니다."""
    store.save(_result(), _meter(), _verdict(), None)
    c = sqlite3.connect(store.path)
    prompt = c.execute("select judge_prompt from verdicts").fetchone()[0]

    for spec in SPECS:
        assert spec.model not in prompt
    assert "참가자 A" in prompt


def test_dropped_participants_are_flagged(store):
    store.save(_result(), _meter(), _verdict(), None)
    c = sqlite3.connect(store.path)
    assert dict(c.execute("select agent_id, dropped from participants")) == {"p1": 0, "p2": 1}


def test_calls_are_queryable_by_purpose_and_round(store):
    store.save(_result(), _meter(), _verdict(), None)
    c = sqlite3.connect(store.path)
    assert dict(c.execute("select purpose, count(*) from llm_calls group by purpose")) \
        == {"debate": 1, "judge": 1}
    assert c.execute("select round_no from llm_calls where purpose='debate'").fetchone()[0] == 1


def test_family_observation_is_recorded_not_enforced(store):
    """막지 않기로 한 결정의 결과물. 나중에 실측 비교를 하려면 기록이 있어야 합니다."""
    store.save(_result(), _meter(), _verdict(),
               family_note("models/gemini-3.5-flash-lite", [s.model for s in SPECS]))
    c = sqlite3.connect(store.path); c.row_factory = sqlite3.Row
    d = c.execute("select * from debates").fetchone()

    assert d["judge_family"] == "gemini"
    assert set(json.loads(d["participant_families"])) == {"gemini", "gpt"}
    assert d["judge_shares_family"] == 1


def test_saving_twice_replaces_rather_than_duplicates(store):
    for _ in range(2):
        store.save(_result(), _meter(), _verdict(), None)
    c = sqlite3.connect(store.path)
    assert c.execute("select count(*) from debates").fetchone()[0] == 1
    assert c.execute("select count(*) from utterances").fetchone()[0] == 2


def test_verdict_is_optional(store):
    store.save(_result(), _meter(), None, None)
    c = sqlite3.connect(store.path)
    assert c.execute("select count(*) from verdicts").fetchone()[0] == 0
    assert c.execute("select count(*) from debates").fetchone()[0] == 1


# ── 사전 견적 ────────────────────────────────────────────────────────────────


def _estimator() -> Estimator:
    return Estimator(PricingTable({
        "m": ModelPrice(Decimal("1.0"), Decimal("2.0")),
    }), ko_tokens_per_char=1.0)


def _specs(n: int) -> list[AgentSpec]:
    return [AgentSpec(f"p{i}", f"L{i}", "prov", "m", "x") for i in range(n)]


def test_estimate_is_a_range_not_a_point():
    """출력 길이를 모르는데 단일 숫자를 내면 그건 거짓말입니다."""
    e = _estimator().estimate(participants=_specs(2), rounds=3, judge_model="m")
    assert e.low_usd < e.high_usd
    assert e.tokens_low < e.tokens_high


def test_estimate_counts_moderator_and_judge_calls():
    e = _estimator().estimate(participants=_specs(3), rounds=3, judge_model="m")
    # 3명×3라운드 + 쟁점추출 1 + 요약 1 + 판정 1
    assert e.calls == 9 + 1 + 1 + 1


def test_estimate_grows_superlinearly_with_participants():
    """참가자를 늘리면 호출 수와 각 호출의 입력이 함께 커집니다. 선형으로
    모델링하면 5명에서 크게 과소 예측합니다."""
    est = _estimator()
    two = est.estimate(participants=_specs(2), rounds=3, judge_model="m").tokens_high
    five = est.estimate(participants=_specs(5), rounds=3, judge_model="m").tokens_high

    assert five > two * 2.5          # 참가자는 2.5배인데 토큰은 그보다 더


def test_unpriced_models_do_not_break_the_estimate():
    e = Estimator(PricingTable({})).estimate(
        participants=_specs(2), rounds=2, judge_model="unknown-model")
    assert e.low_usd == Decimal(0)
    assert "unknown-model" in e.unpriced_models
    assert e.tokens_high > 0         # 토큰은 여전히 셉니다


def test_estimate_reports_the_coefficient_it_used():
    text = _estimator().estimate(participants=_specs(2), rounds=1, judge_model=None).format()
    assert "한국어 토큰 계수 1.0" in text
