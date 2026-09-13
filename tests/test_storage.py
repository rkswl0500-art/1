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


# ── 원장과 토론이 같은 id 를 쓰는가 ─────────────────────────────────────────


def test_ledger_rows_join_back_to_their_debate(store):
    """slice 3 검증이 놓친 것.

    당시 `select purpose, count(*) ... group by purpose` 로만 확인해서 원장이
    **어느 토론 것인지**는 보지 않았습니다. 엔진이 debate_id 를 따로 만들고
    있어서 debates 행과 llm_calls 행의 id 가 서로 달랐고, 조인하는 순간
    한 건도 안 나오는 상태였습니다. group by 는 그걸 못 잡습니다.
    """
    store.save(_result(), _meter(), _verdict(), None)
    c = sqlite3.connect(store.path)

    orphans = c.execute(
        "select count(*) from llm_calls l"
        " where not exists (select 1 from debates d where d.debate_id = l.debate_id)"
    ).fetchone()[0]
    assert orphans == 0

    joined = c.execute(
        "select count(*) from llm_calls l join debates d using (debate_id)"
    ).fetchone()[0]
    assert joined == 2                      # 원장 2건이 토론에 붙어 있어야 함


async def test_engine_uses_the_debate_id_it_was_given():
    """엔진이 id 를 새로 만들면 원장·저장·API 가 서로 다른 값을 갖게 됩니다."""
    from debate.agent import Agent, Anonymizer, Moderator
    from debate.context import ContextBuilder
    from debate.engine import DebateEngine
    from debate.models import DebateConfig
    from debate.provider import FakeBehavior, FakeProvider, MeteredProvider

    specs = [AgentSpec(f"p{i}", f"참가자 {c}", "fake", f"m{i}", "토론자")
             for i, c in ((1, "A"), (2, "B"))]
    meter = CostMeter(PricingTable({}), "d_fixed")
    prov = MeteredProvider(
        FakeProvider({s.model: FakeBehavior(latency_ms=1) for s in specs}), meter)
    anon = Anonymizer(specs)
    engine = DebateEngine([Agent(s, prov, PricingTable({})) for s in specs],
                          ContextBuilder(anon), Moderator(prov, "m1"), anon)

    result = await engine.run(DebateConfig(debate_id="d_fixed", topic="주제",
                                           participants=tuple(specs), rounds=1))

    assert result.debate_id == "d_fixed"
    assert {r.debate_id for r in meter.records} == {"d_fixed"}


def test_sse_stream_honours_last_event_id():
    """서버 계약: id 필드를 붙이고, Last-Event-ID 이후만 보냅니다."""
    import httpx
    from fastapi.testclient import TestClient

    from debate import api

    with TestClient(api.app) as client:
        r = client.post("/debates", json={
            "topic": "T",
            "participants": [{"provider": "fake", "model": "fa"},
                             {"provider": "fake", "model": "fb"}],
            "rounds": 1, "use_fake": True, "gate_timeout_s": 0,
            "fake_latency_ms": {"fa": 1, "fb": 1},
        })
        did = r.json()["debate_id"]
        client.post(f"/debates/{did}/start")

        def read(headers=None):
            ids, types = [], []
            with client.stream("GET", f"/debates/{did}/stream",
                               headers=headers or {}) as s:
                cur = None
                for line in s.iter_lines():
                    if line.startswith("id: "):
                        cur = int(line[4:])
                    elif line.startswith("data: "):
                        import json as _j
                        types.append(_j.loads(line[6:])["type"])
                        ids.append(cur)
                        if types[-1] == "finished":
                            break
            return ids, types

        ids, types = read()
        assert ids and ids == sorted(ids)
        assert ids[0] == 0                         # id 가 seq 와 같음
        assert "finished" in types

        cut = ids[len(ids) // 2]
        ids2, _ = read({"last-event-id": str(cut)})
        assert all(i > cut for i in ids2)          # 커서 이하 재전송 없음

        ids3, _ = read()                           # 헤더 없으면 전체 재생
        assert ids3 == ids


def test_estimate_uses_the_real_header_including_stance_and_persona():
    """고정값으로 두면 입장·페르소나를 넣어 헤더가 길어져도 견적이 반영하지
    못합니다. 효과 자체는 작지만(실측 +19자), 모르는 채로 두는 것과 다릅니다."""
    from debate.models import AgentSpec

    est = _estimator()
    bare = [AgentSpec(f"p{i}", "L", "prov", "m", "토론자") for i in range(2)]
    rich = [AgentSpec(f"p{i}", "L", "prov", "m",
                      "데이터와 측정 방법을 중시하는 실증주의자" * 3, stance="찬성")
            for i in range(2)]

    assert (est.estimate(participants=rich, rounds=2, judge_model=None, topic="주제")
            .tokens_low
            > est.estimate(participants=bare, rounds=2, judge_model=None, topic="주제")
            .tokens_low)


def test_rebuttal_rounds_are_estimated_longer_than_the_opening():
    """R1 은 입론이라 짧고, R2 이후는 반박 구조가 붙어 깁니다. 하나의 넓은
    범위로 뭉개면 정직한 게 아니라 쓸모없어집니다."""
    est = _estimator()
    one = est.estimate(participants=_specs(2), rounds=1, judge_model=None, topic="t")
    two = est.estimate(participants=_specs(2), rounds=2, judge_model=None, topic="t")

    # 라운드가 하나 늘어난 몫이 R1 한 라운드보다 커야 합니다
    assert (two.tokens_high - one.tokens_high) > one.tokens_high


def test_judge_repair_is_reported_apart_from_the_main_range():
    """복구는 일어나거나 안 일어나거나입니다. 본 범위에 섞으면 폭이 1.7배로
    벌어져 '보통 얼마'인지를 잃습니다 — 실측 6,777(복구 없음) 대 9,239(복구 1회)."""
    e = _estimator().estimate(participants=_specs(2), rounds=2,
                              judge_model="m", topic="주제")

    assert e.judge_repair_tokens > 0
    assert e.judge_repair_usd > 0
    # 별도 항목이므로 본 범위에는 들어가 있지 않습니다
    no_judge = _estimator().estimate(participants=_specs(2), rounds=2,
                                     judge_model=None, topic="주제")
    one_judge_share = e.tokens_high - no_judge.tokens_high
    assert e.judge_repair_tokens == one_judge_share


def test_estimate_without_a_judge_has_no_repair_allowance():
    e = _estimator().estimate(participants=_specs(2), rounds=2,
                              judge_model=None, topic="주제")
    assert e.judge_repair_tokens == 0


def test_repair_allowance_is_shown_in_the_formatted_output():
    text = _estimator().estimate(participants=_specs(2), rounds=2,
                                 judge_model="m", topic="주제").format()
    assert "판정 복구가 붙으면" in text


def test_judge_estimate_scales_with_issue_count():
    """판정 출력은 쟁점마다 per_issue 항목이 붙어 길어집니다. 고정값으로 두면
    쟁점이 많은 토론에서 견적이 낮게 나옵니다."""
    est = _estimator()
    three = est.estimate(participants=_specs(2), rounds=2, judge_model="m",
                         topic="주제", issue_count=3)
    five = est.estimate(participants=_specs(2), rounds=2, judge_model="m",
                        topic="주제", issue_count=5)

    assert five.tokens_high > three.tokens_high


def test_judge_estimate_uses_the_same_model_as_the_budget():
    from debate.judge import verdict_content_tokens

    est = _estimator()
    with_judge = est.estimate(participants=_specs(2), rounds=2, judge_model="m",
                              topic="주제", issue_count=4)
    without = est.estimate(participants=_specs(2), rounds=2, judge_model=None,
                           topic="주제", issue_count=4)

    judge_out = with_judge.tokens_high - without.tokens_high
    # 판정 몫 = 입력(전사) + 출력(내용 추정). 출력이 내용 추정과 맞아야 합니다.
    assert judge_out > verdict_content_tokens(4, 2)


def test_pricing_matches_any_vendor_prefix():
    """models/ 만 특별취급하던 것을 일반화했습니다 — Groq 모델은
    openai/gpt-oss-120b 처럼 다른 접두사를 씁니다."""
    from debate.config import ModelPrice, PricingTable

    t = PricingTable({"openai/gpt-oss-120b": ModelPrice(Decimal(1), Decimal(2))})
    assert t.cost_for("openai/gpt-oss-120b", Usage(1_000_000, 0))[1] is True
    assert t.cost_for("gpt-oss-120b", Usage(1_000_000, 0))[1] is True
    assert t.cost_for("groq/openai/gpt-oss-120b", Usage(1_000_000, 0))[1] is True


def test_ambiguous_bare_name_is_reported_unpriced():
    """접두사만 다른 동명 모델이 둘이면 아무거나 고르지 않습니다 — 엉뚱한 단가를
    붙이느니 모른다고 하는 편이 낫습니다."""
    from debate.config import ModelPrice, PricingTable

    t = PricingTable({"a/m": ModelPrice(Decimal(1), Decimal(1)),
                      "b/m": ModelPrice(Decimal(9), Decimal(9))})
    assert t.cost_for("m", Usage(1_000_000, 0)) == (Decimal(0), False)


def test_calibrate_compares_actuals_to_the_model(tmp_path, monkeypatch, capsys):
    """'Groq 심판이 Gemini 보다 길게 쓰는가' 를 추측이 아니라 기록으로 답하려고
    있습니다. 벤더별 상수를 넣을지는 이 숫자가 쌓인 뒤 결정할 문제입니다."""
    from debate.cli import _cmd_calibrate
    from debate.judge import verdict_content_tokens
    from debate.models import AgentSpec, DebateResult, Issue, RoundResult, Usage, Utterance

    db = tmp_path / "debates.db"
    monkeypatch.setenv("DEBATE_DB_PATH", str(db))
    store = SqliteStore(db)
    specs = (AgentSpec("p1", "참가자 A", "g", "m-a", "x"),
             AgentSpec("p2", "참가자 B", "g", "m-b", "x"))
    result = DebateResult(
        debate_id="d1", topic="t", participants=specs,
        rounds=(RoundResult(1, (Utterance("p1", 1, "가" * 100, Usage(10, 70), 5,
                                          Decimal(0)),), 0, 0, 1),),
        issues=(Issue("i1", "a"), Issue("i2", "b"), Issue("i3", "c")))
    meter = CostMeter(PricingTable({}), "d1")
    meter.record(purpose="judge", agent_id=None, provider="groq",
                 model="openai/gpt-oss-120b", usage=Usage(1963, 1504), latency_ms=1)
    store.save(result, meter, None, None)
    store.close()

    assert _cmd_calibrate(object()) == 0
    out = capsys.readouterr().out

    assert "openai/gpt-oss-120b" in out
    modelled = verdict_content_tokens(3, 2)
    assert f"{modelled:,}" in out            # 모델값이 보여야 비교가 됩니다
    assert "1,504" in out                    # 실측값도
    assert "1.36x" in out or "1.35x" in out  # 비율 = 1504 / 1110


def test_calibrate_without_records_is_not_an_error(tmp_path, monkeypatch, capsys):
    from debate.cli import _cmd_calibrate

    db = tmp_path / "empty.db"
    monkeypatch.setenv("DEBATE_DB_PATH", str(db))
    SqliteStore(db).close()

    assert _cmd_calibrate(object()) == 0
    assert "기록이 없습니다" in capsys.readouterr().out + capsys.readouterr().err


def test_judge_output_range_covers_the_measured_underestimate():
    """모델이 체계적으로 낮습니다(실측 1,504 대 1,110). 중앙값을 옮기지 않고
    상한에 여유를 둡니다 — 한 건으로 배수를 확정하면 과적합입니다."""
    from debate.judge import verdict_content_tokens

    est = _estimator()
    with_judge = est.estimate(participants=_specs(2), rounds=2, judge_model="m",
                              topic="주제", issue_count=3)
    without = est.estimate(participants=_specs(2), rounds=2, judge_model=None,
                           topic="주제", issue_count=3)

    judge_share = with_judge.tokens_high - without.tokens_high
    assert judge_share > 1504            # 실측 출력이 상한 안에 들어와야 합니다
    assert int(verdict_content_tokens(3, 2) * 1.5) >= 1504
