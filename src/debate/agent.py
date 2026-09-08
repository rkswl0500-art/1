"""Agent 레이어.

슬라이스 1 은 Agent 하나뿐입니다. Moderator(쟁점 추출/요약)와 Anonymizer
(발언 본문 익명화)는 슬라이스 2 에서 이 파일에 붙습니다.
"""

from __future__ import annotations

from decimal import Decimal

from .config import PricingTable
from .models import AgentSpec, ChatRequest, ContextPack, Usage, Utterance
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
