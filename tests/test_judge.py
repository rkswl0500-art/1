"""Judge: 격리, 파싱 견고성, 편향 대응."""

from __future__ import annotations

import json

import pytest

from debate.judge import Judge, Verdict, family_note, vendor_family
from debate.models import AnonUtterance, ChatResponse, Issue, Usage

LABELS = ["참가자 A", "참가자 B"]
ISSUES = [Issue("i1", "측정 지표"), Issue("i2", "전환 비용")]
TRANSCRIPT = tuple(
    AnonUtterance(label, rnd, f"{label} 의 R{rnd} 발언")
    for rnd in (1, 2) for label in LABELS
)

GOOD = json.dumps({
    "per_issue": [{"issue_id": "i1", "scores": {"참가자 A": 7, "참가자 B": 5},
                   "reasoning": "측정 방법에서 갈림"}],
    "rubric": {"참가자 A": {"근거": 7, "논리": 8, "반박": 6, "명료성": 7},
               "참가자 B": {"근거": 5, "논리": 6, "반박": 7, "명료성": 6}},
    "winner": "참가자 A", "margin": "narrow",
    "conclusion": "A 가 앞섰다", "dissent": "A 는 온보딩 논점에 답하지 않았다",
}, ensure_ascii=False)


class ScriptedProvider:
    """정해진 응답을 순서대로 돌려줍니다."""

    name = "scripted"

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.requests: list = []

    async def chat(self, req):
        self.requests.append(req)
        text = self._replies.pop(0) if self._replies else "{}"
        return ChatResponse(text=text, model="judge-model",
                            usage=Usage(100, 50), latency_ms=1, finish_reason="stop")

    async def list_models(self): return []
    async def aclose(self): return None


async def _judge(*replies: str) -> tuple[Verdict, ScriptedProvider]:
    p = ScriptedProvider(*replies)
    v = await Judge(p, "judge-model").evaluate("주제", ISSUES, TRANSCRIPT, LABELS)
    return v, p


# ── 파싱 ─────────────────────────────────────────────────────────────────────


async def test_parses_a_well_formed_verdict():
    v, _ = await _judge(GOOD)
    assert v.status == "ok"
    assert v.winner == "참가자 A"
    assert v.totals() == {"참가자 A": 28, "참가자 B": 24}
    assert v.dissent


async def test_strips_code_fences_and_prose():
    v, _ = await _judge(f"판정입니다:\n```json\n{GOOD}\n```\n이상입니다.")
    assert v.status == "ok" and v.winner == "참가자 A"


async def test_repairs_once_then_succeeds():
    v, provider = await _judge("이건 JSON 이 아닙니다", GOOD)
    assert v.status == "ok"
    assert v.parse_attempts == 2
    assert len(provider.requests) == 2
    # 복구 요청에 무엇이 틀렸는지가 실려야 합니다
    assert "파싱되지 않았습니다" in provider.requests[1].messages[-1].content


async def test_two_failures_preserve_the_transcript_not_crash():
    """Judge 가 깨져도 토론 기록은 살아야 합니다."""
    v, _ = await _judge("망가짐", "여전히 망가짐")
    assert v.status == "unparsed"
    assert v.raw == "여전히 망가짐"
    assert v.winner is None
    assert v.prompt_text                      # 무엇을 보냈는지는 남음


async def test_unknown_winner_label_is_rejected():
    bad = json.dumps({"rubric": {"참가자 A": {"근거": 5, "논리": 5, "반박": 5, "명료성": 5}},
                      "winner": "참가자 Z"}, ensure_ascii=False)
    v, provider = await _judge(bad, GOOD)
    assert v.status == "ok"                   # 복구 패스가 살림
    assert "참가자 Z" in provider.requests[1].messages[-1].content


async def test_scores_are_clamped_to_the_rubric_range():
    wild = json.dumps({
        "rubric": {"참가자 A": {"근거": 99, "논리": -5, "반박": "많음", "명료성": 7},
                   "참가자 B": {"근거": 5, "논리": 5, "반박": 5, "명료성": 5}},
        "winner": None, "margin": "tie", "conclusion": "", "dissent": "",
    }, ensure_ascii=False)
    v, _ = await _judge(wild)
    # 빠진 축(일관성)은 0 으로 채워집니다 — 키가 없다고 터지면 안 됩니다.
    assert v.rubric["참가자 A"] == {"근거": 10, "논리": 0, "반박": 0,
                                  "명료성": 7, "일관성": 0}


# ── 편향 대응 ────────────────────────────────────────────────────────────────


async def test_judge_prompt_carries_labels_not_models():
    v, _ = await _judge(GOOD)
    assert "참가자 A" in v.prompt_text
    for leak in ("gpt", "gemini", "claude", "judge-model"):
        assert leak not in v.prompt_text.lower()


async def test_presentation_order_is_shuffled_across_seeds():
    """위치 편향 대응. 순서가 고정이면 매번 같은 쪽이 유리합니다."""
    def order(seed: int) -> list[str]:
        rendered = Judge(ScriptedProvider(), "m", seed=seed)._render_transcript(TRANSCRIPT)
        return [ln.rstrip(":") for ln in rendered.splitlines() if ln.startswith("참가자")]

    orders = {tuple(order(s)) for s in range(8)}
    assert len(orders) > 1                    # 시드에 따라 순서가 달라짐


async def test_every_utterance_survives_the_shuffle():
    rendered = Judge(ScriptedProvider(), "m", seed=3)._render_transcript(TRANSCRIPT)
    for u in TRANSCRIPT:
        assert u.content in rendered
    assert rendered.index("제1라운드") < rendered.index("제2라운드")   # 라운드 순서는 유지


async def test_truncated_verdict_is_observable():
    """단일 패스를 언제 갈라야 할지의 신호. 막지 않고 관측만 합니다."""
    class Truncating(ScriptedProvider):
        async def chat(self, req):
            from dataclasses import replace
            return replace(await super().chat(req), finish_reason="length")

    v = await Judge(Truncating(GOOD), "m").evaluate("주제", ISSUES, TRANSCRIPT, LABELS)
    assert v.truncated
    assert v.prompt_tokens == 100


# ── 계열 관측 (막지 않음) ────────────────────────────────────────────────────


@pytest.mark.parametrize("model,family", [
    ("models/gemini-3.6-flash", "gemini"), ("gpt-4o-mini", "gpt"),
    ("llama-3.3-70b", "llama"), ("claude-sonnet-5", "claude"),
])
def test_vendor_family_extraction(model, family):
    assert vendor_family(model) == family


def test_family_note_states_a_fact_without_blocking():
    same = family_note("models/gemini-3.5-flash-lite", ["models/gemini-3.6-flash"])
    assert same.shares_family
    assert "참가자와 동일" in same.line()

    diff = family_note("gpt-4o-mini", ["models/gemini-3.6-flash"])
    assert not diff.shares_family
    assert "참가자와 다름" in diff.line()
    # 사실 진술이지 경고가 아닙니다
    assert "경고" not in diff.line() and "주의" not in diff.line()


# ── 출력 예산과 잘림 (라이브에서 드러남) ────────────────────────────────────


def test_budget_scales_with_issues_and_participants():
    """쟁점·참가자가 늘면 판정 JSON 도 길어집니다. 고정값이면 큰 토론에서 잘립니다."""
    from debate.judge import required_output_tokens as need

    assert need(5, 2) > need(3, 2)
    assert need(3, 5) > need(3, 2)


def test_budget_reserves_a_fixed_share_for_thinking():
    """사고 몫은 모델 속성이지 쟁점 수의 함수가 아닙니다. 배수로 처리하면
    쟁점이 적을 때 모자라고 많을 때 과합니다."""
    from debate.judge import required_output_tokens as need

    thinking = need(4, 2) - need(4, 2, thinking_reserve=0)
    assert thinking == 6000
    # 사고 몫을 뺀 내용 몫은 라이브에서 관측된 판정 JSON 크기(~1,000 토큰) 수준
    assert 1000 <= need(4, 2, thinking_reserve=0) <= 3000


def test_default_budget_exceeds_the_size_that_truncated_live():
    """라이브에서 4,096 으로 JSON 이 중간에 끊겼습니다."""
    from debate.judge import required_output_tokens as need

    assert need(4, 2) > 4096


async def test_truncated_verdict_is_labelled_truncated_not_malformed():
    """둘을 '파싱 실패'로 묶으면 예산 문제를 프롬프트 문제로 오해합니다."""
    class Truncating(ScriptedProvider):
        async def chat(self, req):
            from dataclasses import replace
            return replace(await super().chat(req), finish_reason="length")

    v = await Judge(Truncating('{"per_issue": [{"issue_id": "i1"',
                               '{"per_issue": [{"issue_id": "i1"'),
                    "m").evaluate("주제", ISSUES, TRANSCRIPT, LABELS)

    assert v.status == "unparsed"
    assert v.failure_kind == "truncated"
    assert v.finish_reason == "length"


async def test_malformed_verdict_is_labelled_malformed():
    v, _ = await _judge("이건 JSON 이 아닙니다", "여전히 아닙니다")
    assert v.status == "unparsed"
    assert v.failure_kind == "malformed"


async def test_length_failure_raises_the_budget_and_asks_for_brevity():
    """같은 예산으로 다시 부르면 같은 자리에서 또 잘립니다. 예산만 올리면
    장황한 모델은 늘어난 만큼 더 씁니다 — 둘 다 해야 합니다."""
    budgets: list[int] = []

    class Truncating(ScriptedProvider):
        async def chat(self, req):
            from dataclasses import replace
            budgets.append(req.max_tokens)
            resp = await super().chat(req)
            return replace(resp, finish_reason="length" if len(budgets) == 1 else "stop")

    provider = Truncating('{"per_issue": [{"issue_id"', GOOD)
    v = await Judge(provider, "m").evaluate("주제", ISSUES, TRANSCRIPT, LABELS)

    assert v.status == "ok"                       # 복구 성공
    assert budgets[1] == budgets[0] * 2           # 예산 2배
    repair = provider.requests[1].messages[-1].content
    assert "한 문장" in repair                     # 동시에 더 짧게 지시
    # 잘린 본문을 되돌려 보내지 않습니다 — 예산만 먹습니다
    assert not any(m.role == "assistant" for m in provider.requests[1].messages)


def test_prompt_caps_reasoning_length():
    """쟁점당 2~3문장이면 충분하고, 길수록 잘릴 위험만 커집니다."""
    from debate.judge import REASONING_CHARS, _PROMPT

    rendered = _PROMPT.format(topic="t", issues="i", reasoning_chars=REASONING_CHARS,
                             rubric_example='"근거": 7')
    assert f"{REASONING_CHARS}자 이내" in rendered
    assert "JSON 을 반드시 닫으십시오" in rendered
    assert 80 <= REASONING_CHARS <= 200


# ── 일관성 축 (2회차 라이브에서 드러남) ─────────────────────────────────────


def test_rubric_scores_self_consistency():
    """4축만으로는 라운드마다 입장을 갈아타는 참가자를 못 잡습니다. 실제로
    R1 주장을 R2 에서 스스로 반박한 참가자들이 반박 9/8점을 받았습니다 —
    형식적으로는 반박 구조를 갖췄으니 그 점수가 맞고, 그래서 축이 더 필요합니다."""
    from debate.judge import RUBRIC_KEYS, REASONING_CHARS, _PROMPT

    assert "일관성" in RUBRIC_KEYS

    rendered = _PROMPT.format(topic="t", issues="i", reasoning_chars=REASONING_CHARS,
                             rubric_example='"근거": 7')
    assert "자기 입장을 유지했는가" in rendered
    assert "스스로 반박하거나 슬그머니 뒤집으면" in rendered
    # 설득당해 바꾼 것은 감점하지 않습니다 — 그건 토론이 작동한 것입니다
    assert "무엇이 자신을 설득했는지 밝히고" in rendered


async def test_consistency_score_is_parsed_and_totalled():
    good = json.dumps({
        "per_issue": [],
        "rubric": {"참가자 A": {"근거": 7, "논리": 7, "반박": 9, "명료성": 7, "일관성": 2},
                   "참가자 B": {"근거": 6, "논리": 6, "반박": 8, "명료성": 6, "일관성": 3}},
        "winner": "참가자 B", "margin": "narrow", "conclusion": "c", "dissent": "d",
    }, ensure_ascii=False)
    v, _ = await _judge(good)

    assert v.rubric["참가자 A"]["일관성"] == 2
    assert v.totals() == {"참가자 A": 32, "참가자 B": 29}


def test_budget_grows_with_the_number_of_rubric_axes():
    """축을 늘리면 판정 JSON 도 길어집니다. 상수로 박아두면 축 추가가
    조용히 잘림을 부릅니다."""
    from debate.judge import RUBRIC_KEYS, required_output_tokens as need

    assert need(4, 2, thinking_reserve=0) > 15 * len(RUBRIC_KEYS) * 2


def test_no_module_hardcodes_the_rubric_axes():
    """UI 표·CLI 출력·fake 판정 세 곳이 각자 축 이름을 적고 있었습니다.
    '일관성' 을 추가했을 때 전부 조용히 어긋날 뻔했습니다."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "debate"
    literal = '"근거"'
    offenders = [
        f.name for f in src.rglob("*.py")
        if f.name != "models.py" and literal in f.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"축을 하드코딩한 모듈: {offenders}"

    html = (src / "static" / "index.html").read_text(encoding="utf-8")
    assert "'근거','논리'" not in html      # 표는 판정 데이터에서 도출합니다


async def test_fake_judge_covers_every_rubric_axis():
    """fake 가 축을 빠뜨리면 0 으로 채워져 루브릭 검증이 무의미해집니다."""
    from debate.models import RUBRIC_KEYS
    from debate.provider import FakeProvider
    from debate.models import ChatRequest, Message

    resp = await FakeProvider().chat(ChatRequest(
        model="m", messages=(Message("user", "참가자 A 참가자 B"),), purpose="judge"))
    rubric = json.loads(resp.text)["rubric"]

    for label, scores in rubric.items():
        assert set(scores) == set(RUBRIC_KEYS), f"{label}: {sorted(scores)}"


@pytest.mark.parametrize("model,family", [
    ("models/gemini-3.6-flash", "gemini"),
    ("gemini-3.5-flash-lite", "gemini"),
    ("openai/gpt-oss-120b", "gpt"),
    ("gpt-4o-mini", "gpt"),
    ("llama-3.3-70b:free", "llama"),
    ("claude-sonnet-5", "claude"),
])
def test_vendor_family_matches_what_the_ui_computes(model, family):
    """UI 가 고르는 시점에 같은 계열인지 알려주려고 클라이언트에서도 같은 판정을
    합니다(index.html 의 vendorOf). 두 구현이 어긋나면 화면과 기록이 다른 말을
    하므로, 서버 쪽을 바꾸면 그쪽도 같이 고쳐야 합니다."""
    assert vendor_family(model) == family


def test_verdict_size_has_one_definition():
    """예산(judge)과 견적(cost)이 각자 추정하면 한쪽만 갱신되어 어긋납니다.
    실제로 견적 쪽은 쟁점 수를 아예 반영하지 않는 고정값이었습니다."""
    from debate.judge import required_output_tokens, verdict_content_tokens

    assert verdict_content_tokens(5, 3) > verdict_content_tokens(3, 3)
    assert verdict_content_tokens(3, 5) > verdict_content_tokens(3, 3)
    # 예산 = 사고 몫 + 내용 몫 × 1.5
    assert required_output_tokens(4, 2, thinking_reserve=0) == int(
        verdict_content_tokens(4, 2) * 1.5)


# ── 산문의 점수가 표와 어긋나는 경우 (라이브: 합계 40 을 "23점" 이라 씀) ────


def _verdict(conclusion: str = "", dissent: str = "") -> Verdict:
    return Verdict(
        per_issue=(),
        rubric={"참가자 A": {"근거": 7, "논리": 7, "반박": 6, "명료성": 7, "일관성": 6},
                "참가자 B": {"근거": 8, "논리": 8, "반박": 8, "명료성": 8, "일관성": 8}},
        winner="참가자 B", margin="narrow", conclusion=conclusion, dissent=dissent)


def test_wrong_total_in_the_conclusion_is_caught():
    """모델이 자기 JSON 을 보고 합계를 다시 계산하다 틀립니다. 점수 필드는
    정확하지만, 사용자가 결론만 읽으면 틀린 숫자를 믿습니다."""
    from debate.judge import score_contradictions

    notes = score_contradictions(_verdict("참가자 B가 23점으로 소폭 앞섰다"))

    assert len(notes) == 1
    assert "23점" in notes[0]
    assert "참가자 B 40" in notes[0]        # 실제 합계를 같이 보여줍니다


def test_correct_total_in_the_conclusion_passes():
    from debate.judge import score_contradictions

    assert score_contradictions(_verdict("참가자 B가 40점으로 앞섰다")) == ()


@pytest.mark.parametrize("text", [
    "B가 8점대로 고르게 높았다",              # 축 점수 범위 — 합계 주장이 아님
    "스탠퍼드 13% 생산성 향상 연구를 인용했다",   # 백분율
    "2020년 사례를 들었다",                   # 연도
    "B가 근거의 구체성에서 앞섰다",             # 숫자 없음
])
def test_no_false_positives_on_ordinary_prose(text):
    """오탐이 나면 경고가 노이즈가 되고, 노이즈가 되면 진짜도 무시됩니다."""
    from debate.judge import score_contradictions

    assert score_contradictions(_verdict(text)) == ()


def test_dissent_is_checked_too():
    from debate.judge import score_contradictions

    notes = score_contradictions(_verdict("", "A 는 99점을 받을 만한 반론에 답하지 않았다"))
    assert len(notes) == 1
    assert "미해결 반론" in notes[0]


def test_unparsed_verdict_is_not_checked():
    """파싱 실패면 비교할 합계 자체가 없습니다."""
    from dataclasses import replace
    from debate.judge import score_contradictions

    broken = replace(_verdict("23점"), status="unparsed", rubric={})
    assert score_contradictions(broken) == ()


def test_prompt_forbids_computing_totals_in_prose():
    from debate.judge import REASONING_CHARS, _PROMPT
    from debate.models import RUBRIC_KEYS

    rendered = _PROMPT.format(topic="t", issues="i", reasoning_chars=REASONING_CHARS,
                              rubric_example=", ".join(f'"{k}": 7' for k in RUBRIC_KEYS))
    assert "합계 점수를 쓰지 마십시오" in rendered
