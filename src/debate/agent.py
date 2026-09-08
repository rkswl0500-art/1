"""Agent 레이어.

슬라이스 1 은 Agent 하나뿐입니다. Moderator(쟁점 추출/요약)와 Anonymizer
(발언 본문 익명화)는 슬라이스 2 에서 이 파일에 붙습니다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from .config import PricingTable
from .models import (
    AgentSpec, AnonUtterance, ChatRequest, ContextPack, Issue, Message,
    Usage, Utterance,
)
from .provider import ChatProvider, ProviderError

#: 모든 참가자에게 공통으로 붙는 규칙. 마지막 항목이 익명화의 1차 방어선입니다
#: (2차는 슬라이스 2 의 Anonymizer.scrub). 문체 기반 추정까지는 못 막습니다.
DEBATE_RULES = """[규칙]
- 한국어로 답하십시오.
- 근거를 먼저 제시하고 주장을 뒤에 두십시오.
- 상대를 인신공격하지 말고 주장만 다투십시오.
- 600자 이내로 쓰십시오.
- 당신이 어떤 모델인지, 어느 회사가 만들었는지 절대 언급하지 마십시오.
  자기소개, 서명, "AI로서" 같은 표현도 쓰지 마십시오."""


def build_header(spec: AgentSpec, topic: str) -> str:
    """참가자별 고정 헤더. 라운드가 바뀌어도 이 부분은 안 바뀝니다."""
    lines = [
        f"당신은 토론 참가자 «{spec.label}» 입니다.",
        f"[주제] {topic}",
        f"[당신의 역할] {spec.persona}",
    ]
    if spec.stance:
        lines.append(f"[당신의 입장] {spec.stance}")
    lines.append(DEBATE_RULES)
    return "\n\n".join(lines)


class Agent:
    """참가자 한 명. 컨텍스트를 받아 발언 하나를 만듭니다.

    얇게 유지합니다 — 프롬프트 조립은 ContextPack 이, 재시도/계측은 프로바이더
    데코레이터가 합니다. 여기서 하는 일은 호출과 Utterance 포장뿐입니다.
    """

    def __init__(
        self,
        spec: AgentSpec,
        provider: ChatProvider,
        pricing: PricingTable,
        *,
        max_tokens: int | None = 2048,
        timeout_s: float = 120.0,
    ) -> None:
        self.spec = spec
        self._provider = provider
        self._pricing = pricing
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def label(self) -> str:
        return self.spec.label

    async def speak(self, ctx: ContextPack) -> Utterance:
        req = ChatRequest(
            model=self.spec.model,
            messages=ctx.render(),
            temperature=self.spec.temperature,
            max_tokens=self._max_tokens,
            timeout_s=self._timeout_s,
            tag=self.spec.id,
            purpose="debate",
            round_no=ctx.round_no,
        )
        resp = await self._provider.chat(req)  # 실패는 그대로 위로 — 엔진이 판단
        cost, _ = self._pricing.cost_for(resp.model, resp.usage)
        return Utterance(
            agent_id=self.spec.id,
            round_no=ctx.round_no,
            content=resp.text.strip(),
            usage=resp.usage,
            latency_ms=resp.latency_ms,
            cost_usd=cost,
            finish_reason=resp.finish_reason,
        )

    @staticmethod
    def failed(spec: AgentSpec, round_no: int, err: BaseException) -> Utterance:
        return Utterance(
            agent_id=spec.id,
            round_no=round_no,
            content="",
            usage=Usage(0, 0),
            latency_ms=0,
            cost_usd=Decimal(0),
            status="failed",
            error=f"{type(err).__name__}: {err}",
        )


# ── 익명화 ───────────────────────────────────────────────────────────────────

REDACTED = "[비공개]"

#: 자기소개 문맥에서만 위험한 단어들. 이 목록에 있다는 이유만으로 무조건
#: 지우지는 않습니다 — 주제가 "구글의 원격근무 정책"이면 'Google' 은 논거이지
#: 정체 누설이 아닙니다.
_VENDOR_WORDS = (
    "openai", "gpt", "chatgpt", "anthropic", "claude", "google", "gemini",
    "deepmind", "meta", "llama", "mistral", "cohere", "qwen", "alibaba",
    "구글", "오픈에이아이", "앤트로픽", "제미나이", "메타",
)

#: 1인칭 자기 정체 선언. 이건 문맥과 무관하게 누설입니다.
_SELF_ID_PATTERNS = (
    re.compile(r"(저는|제가|나는|본인은)[^.!?\n]{0,40}?(" + "|".join(_VENDOR_WORDS) + r")"
               r"[^.!?\n]{0,40}?(입니다|이며|로서|만든|개발한|훈련|기반)", re.I),
    re.compile(r"(" + "|".join(_VENDOR_WORDS) + r")\s*(이|가|에서)?\s*"
               r"(만든|개발한|훈련시킨|제작한)", re.I),
    re.compile(r"as an? (ai )?(language )?model (developed|created|trained|made) by [^.!?\n]+", re.I),
    re.compile(r"(저는|제가)\s*(하나의\s*)?(AI|인공지능)(로서|으로서|입니다|이다)", re.I),
)


@dataclass(frozen=True, slots=True)
class ScrubResult:
    text: str
    redactions: int
    #: 지우지는 않았지만 벤더 단어가 남아 있는 곳. 사용자 검토용.
    residual_vendor_words: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.redactions > 0


class Anonymizer:
    """모델 정체를 컨텍스트와 Judge 프롬프트에서 걷어냅니다.

    **완전한 방어가 아닙니다.** 문체만으로 추정하는 건 막을 수 없고, 그래서
    이건 방어선 하나이지 보증이 아닙니다. 프롬프트의 자기소개 금지 규칙이
    1차, 여기가 2차입니다.

    보수적으로 지웁니다. 참가자의 모델 ID·프로바이더 이름은 순수 메타데이터라
    항상 지우지만, 벤더 단어가 그냥 등장하는 건 논거일 수 있어 남겨두고 세기만
    합니다 — 토론 내용을 파괴하는 익명화는 익명화가 아니라 검열입니다.
    """

    def __init__(self, specs: Sequence[AgentSpec]) -> None:
        self._labels = {s.id: s.label for s in specs}
        tokens: set[str] = set()
        for spec in specs:
            tokens.add(spec.model)
            tokens.add(spec.model.rsplit("/", 1)[-1])   # models/x-1.0 -> x-1.0
            tokens.add(spec.provider)
        # 긴 것부터 지워야 부분 문자열이 먼저 지워지는 일이 없습니다.
        self._tokens = tuple(sorted((t for t in tokens if len(t) > 2), key=len, reverse=True))

    def label_of(self, agent_id: str) -> str:
        return self._labels.get(agent_id, agent_id)

    def scrub(self, text: str) -> ScrubResult:
        """컨텍스트에 들어가는 **모든** 텍스트에 적용합니다.

        참가자 발언뿐 아니라 사람이 넣는 지시문에도 씁니다 — 사용자는 어느
        참가자가 어느 모델인지 알기 때문에, 지시문에 모델명을 쓰면 그게 그대로
        컨텍스트를 거쳐 Judge 프롬프트까지 흘러갑니다.
        """
        out, hits = text, 0
        for token in self._tokens:
            if token and token.lower() in out.lower():
                out, n = re.subn(re.escape(token), REDACTED, out, flags=re.I)
                hits += n
        for pattern in _SELF_ID_PATTERNS:
            out, n = pattern.subn(REDACTED, out)
            hits += n
        residual = tuple(sorted({
            w for w in _VENDOR_WORDS if re.search(re.escape(w), out, re.I)
        }))
        return ScrubResult(text=out, redactions=hits, residual_vendor_words=residual)

    def to_anon(self, utterance: Utterance) -> AnonUtterance:
        return AnonUtterance(
            label=self.label_of(utterance.agent_id),
            round_no=utterance.round_no,
            content=self.scrub(utterance.content).text,
        )


# ── 사회자 ───────────────────────────────────────────────────────────────────

MIN_ISSUES, MAX_ISSUES = 3, 5

_ISSUE_PROMPT = """당신은 토론 사회자입니다. 참가자가 아니며 어느 편도 들지 않습니다.

아래는 «{topic}» 에 대한 1라운드 발언 전문입니다.
이후 라운드가 흩어지지 않도록 **핵심 쟁점 {lo}~{hi}개**를 뽑으십시오.

- 양측이 실제로 충돌하는 지점만 고르십시오. 한쪽만 말한 것은 쟁점이 아닙니다.
- 각 쟁점은 한 문장의 한국어 명사구 또는 의문문으로 쓰십시오.
- 다른 말 없이 아래 JSON 만 출력하십시오.

{{"issues": ["...", "..."]}}"""

_SUMMARY_PROMPT = """당신은 토론 사회자입니다. 아래 라운드를 압축하십시오.

이후 라운드의 참가자들은 직전 라운드는 전문으로 보지만, 그 이전은 이 요약으로만
봅니다. 따라서 **누가 무엇을 주장했고 무엇이 반박됐는지**가 남아야 합니다.

- 400자 이내, 한국어.
- 참가자 라벨(참가자 A 등)은 그대로 유지하십시오.
- 새로운 의견을 보태지 말고 있는 내용만 압축하십시오.
{trace_rule}
[기존 요약]
{prior}

[이번에 압축할 라운드]
{round_text}"""


class Moderator:
    """참가자도 Judge 도 아닌 제3의 역할.

    쟁점 추출을 참가자에게 시키면 자기 논점만 쟁점으로 올리고, Judge 에게
    시키면 전체 토론을 읽기 전에 선입견이 생깁니다. 그래서 분리했습니다.
    Judge 는 슬라이스 3 에서 순수 평가만 맡습니다.
    """

    def __init__(self, provider, model: str, *, max_tokens: int = 1024,
                 timeout_s: float = 120.0) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s

    async def extract_issues(
        self, topic: str, round1: Sequence[AnonUtterance]
    ) -> tuple[Issue, ...]:
        prompt = _ISSUE_PROMPT.format(topic=topic, lo=MIN_ISSUES, hi=MAX_ISSUES)
        body = "\n\n".join(f"{u.label}:\n{u.content}" for u in round1)
        resp = await self._provider.chat(ChatRequest(
            model=self._model,
            messages=(Message("system", prompt), Message("user", body)),
            temperature=0.2, max_tokens=self._max_tokens,
            timeout_s=self._timeout_s, purpose="issues", round_no=1,
        ))
        titles = _parse_issues(resp.text)
        return tuple(Issue(id=f"i{n}", title=t) for n, t in enumerate(titles, start=1))

    async def summarize(
        self, prior: str, round_result, anonymize, *,
        directive_trace: Sequence[tuple[int, str]] = (),
    ) -> str:
        round_text = "\n\n".join(
            f"{a.label}:\n{a.content}"
            for a in (anonymize(u) for u in round_result.utterances if u.status == "ok")
        )
        trace_rule = ""
        if directive_trace:
            noted = "; ".join(f"R{n}: {t}" for n, t in directive_trace)
            # 지시는 1회용이지만 "왜 갑자기 이 얘기가 나왔는지"는 남아야 합니다.
            trace_rule = (
                f"- 아래 사회자 지시가 있었다는 사실을 한 줄로 남기십시오: {noted}\n"
            )
        resp = await self._provider.chat(ChatRequest(
            model=self._model,
            messages=(Message("system", _SUMMARY_PROMPT.format(
                trace_rule=trace_rule, prior=prior or "(없음)", round_text=round_text)),),
            temperature=0.2, max_tokens=self._max_tokens,
            timeout_s=self._timeout_s, purpose="summary",
            round_no=round_result.round_no,
        ))
        return resp.text.strip()


def _parse_issues(raw: str) -> tuple[str, ...]:
    """JSON 우선, 실패하면 번호 목록. 둘 다 실패하면 예외.

    쟁점 추출이 조용히 빈 목록을 돌려주면 이후 라운드가 아무 제약 없이 흩어지고,
    그건 실패인데 성공처럼 보입니다.
    """
    titles: list[str] = []
    text = raw.strip()
    if "```" in text:                      # ```json 펜스 제거
        text = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.M).strip()

    try:
        data = json.loads(text)
        raw_items = data["issues"] if isinstance(data, dict) else data
        for item in raw_items:
            titles.append(item["title"] if isinstance(item, dict) else str(item))
    except (ValueError, KeyError, TypeError):
        for line in text.splitlines():          # 폴백: "1) ..." / "- ..."
            m = re.match(r"\s*(?:\d+[).\]]|[-*•])\s*(.+)", line)
            if m:
                titles.append(m.group(1).strip())

    titles = [t.strip() for t in titles if t and t.strip()]
    if len(titles) < MIN_ISSUES:
        raise ValueError(
            f"쟁점을 {MIN_ISSUES}개 이상 뽑지 못했습니다 (얻은 것: {len(titles)}개). "
            f"사회자 응답: {raw[:200]!r}"
        )
    return tuple(titles[:MAX_ISSUES])
