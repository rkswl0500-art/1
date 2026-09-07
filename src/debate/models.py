"""전 레이어 공용 타입.

여기에는 동작이 거의 없습니다. 각 레이어가 서로를 import 하지 않고도 같은
어휘를 쓰게 하려고 존재합니다. 의존 방향은 한쪽입니다:

    models <- config <- cost <- provider <- agent <- engine <- cli
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, Union

# ── Provider 어휘 ────────────────────────────────────────────────────────────

Role = Literal["system", "user", "assistant"]

#: LLM 호출의 용도. 비용 원장에서 토론/쟁점추출/요약/판정을 구분합니다.
CallPurpose = Literal["debate", "issues", "summary", "judge"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout_s: float = 120.0
    #: 원장 귀속용 자유 태그(보통 agent_id). 프로바이더는 무시합니다.
    tag: str | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    #: 실제로 서빙된 모델. 요청한 모델과 다를 수 있어 따로 기록합니다.
    model: str
    usage: Usage
    #: 재시도를 포함한 총 소요 시간(RetryingProvider 가 덮어씁니다).
    latency_ms: int
    finish_reason: str | None = None
    #: 성공까지 걸린 시도 횟수. 1 이면 첫 시도에 성공.
    attempts: int = 1


# ── Agent 어휘 ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AgentSpec:
    id: str
    #: 외부/Judge 노출용 익명 라벨. 모델명은 여기 절대 안 들어갑니다.
    label: str
    #: .env 프로바이더 슬롯의 NAME. 참가자마다 다른 엔드포인트를 쓸 수 있습니다.
    provider: str
    model: str
    persona: str
    temperature: float = 0.7
    stance: str | None = None


@dataclass(frozen=True, slots=True)
class Utterance:
    agent_id: str
    round_no: int
    content: str
    usage: Usage
    latency_ms: int
    cost_usd: Decimal
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AnonUtterance:
    """익명화된 발언. Judge 와 타 참가자에게는 이 형태로만 전달됩니다."""

    label: str
    round_no: int
    content: str


@dataclass(frozen=True, slots=True)
class Issue:
    id: str
    title: str


# ── Context ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ContextPack:
    """참가자 한 명에게 보낼 컨텍스트.

    구조가 곧 규칙입니다. 진행 중인 라운드를 담을 필드가 아예 없으므로
    "같은 라운드 참가자는 서로 못 본다"가 프롬프트 훈계가 아니라 타입으로
    보장됩니다. 이 팩을 만드는 ContextBuilder 는 슬라이스 2에서 붙습니다.
    """

    header: str
    round_no: int
    issues: tuple[Issue, ...] = ()
    prior_digest: str = ""
    last_round: tuple[AnonUtterance, ...] = ()

    def render(self) -> tuple[Message, ...]:
        parts: list[str] = []

        if self.issues:
            listed = "\n".join(f"  {i + 1}) {x.title}" for i, x in enumerate(self.issues))
            parts.append(f"[확정된 쟁점] 아래 범위 안에서만 논쟁하십시오.\n{listed}")

        if self.prior_digest:
            parts.append(f"[이전 라운드 요약]\n{self.prior_digest}")

        if self.last_round:
            said = "\n\n".join(f"{u.label}:\n{u.content}" for u in self.last_round)
            parts.append(f"[직전 라운드 발언 전문]\n{said}")

        parts.append(
            f"[당신의 차례] 제{self.round_no}라운드 발언을 작성하십시오."
            if self.round_no > 1
            else "[당신의 차례] 제1라운드 입론을 작성하십시오."
        )

        return (Message("system", self.header), Message("user", "\n\n".join(parts)))


# ── Engine ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DebateConfig:
    topic: str
    participants: tuple[AgentSpec, ...]
    rounds: int = 3
    max_concurrency: int = 3
    round_timeout_s: float = 180.0


@dataclass(frozen=True, slots=True)
class RoundResult:
    round_no: int
    utterances: tuple[Utterance, ...]
    #: 라운드 전체 벽시계 시간.
    wall_ms: int
    #: 개별 지연의 단순 합. wall_ms 와 비교하면 병렬 여부가 눈에 보입니다.
    sum_latency_ms: int
    #: ceil(참가자수 / 동시성). 세마포어 때문에 wall 이 max 보다 큰 이유.
    waves: int


DebateStatus = Literal["completed", "aborted_insufficient_participants"]


@dataclass(frozen=True, slots=True)
class DebateResult:
    debate_id: str
    topic: str
    participants: tuple[AgentSpec, ...]
    rounds: tuple[RoundResult, ...]
    status: DebateStatus = "completed"
    dropped: tuple[str, ...] = ()


# ── 이벤트 ───────────────────────────────────────────────────────────────────
# Storage(슬라이스 3)와 SSE(슬라이스 4)가 각각 구독합니다. 엔진은 구독자를
# 모릅니다. 슬라이스 2/3 에서 IssuesExtracted 등이 이 union 에 추가됩니다.


@dataclass(frozen=True, slots=True)
class DebateStarted:
    debate_id: str
    topic: str
    participants: tuple[AgentSpec, ...]
    rounds: int


@dataclass(frozen=True, slots=True)
class RoundStarted:
    round_no: int
    active: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UtteranceCompleted:
    utterance: Utterance
    label: str


@dataclass(frozen=True, slots=True)
class AgentDropped:
    agent_id: str
    round_no: int
    reason: str


@dataclass(frozen=True, slots=True)
class RoundCompleted:
    result: RoundResult


@dataclass(frozen=True, slots=True)
class DebateCompleted:
    result: DebateResult


DebateEvent = Union[
    DebateStarted,
    RoundStarted,
    UtteranceCompleted,
    AgentDropped,
    RoundCompleted,
    DebateCompleted,
]


@dataclass(frozen=True, slots=True)
class CallRecord:
    """비용 원장 1행. 모든 LLM 호출이 용도 불문 여기에 남습니다."""

    call_id: str
    debate_id: str
    purpose: CallPurpose
    agent_id: str | None
    provider: str
    model: str
    usage: Usage
    latency_ms: int
    cost_usd: Decimal
    priced: bool
    attempts: int = 1
    finish_reason: str | None = None


class ConfigError(Exception):
    """설정이 잘못됨. LLM 을 한 번도 부르기 전에 던집니다."""
