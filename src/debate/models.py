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
    #: 이 호출의 용도. 같은 프로바이더 인스턴스를 토론/쟁점/요약이 공유하므로
    #: 용도는 호출마다 실려야 합니다 — 생성 시점에 고정하면 사회자 호출이
    #: 전부 'debate' 로 기록됩니다.
    purpose: CallPurpose = "debate"
    #: 몇 번째 라운드의 호출인지. 원장 조회와 fake 장애 주입에 씁니다.
    round_no: int | None = None


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
    #: 'length' 면 모델이 예산을 다 써 잘렸다는 뜻입니다. 토론에서는 치명적이라
    #: 기록만 하지 말고 반드시 사용자에게 보여야 합니다.
    finish_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


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


@dataclass(frozen=True, slots=True)
class Directive:
    """토론 중간에 사람이 넣는 지시.

    **1회용입니다.** 지정된 라운드의 컨텍스트에만 들어가고, 그 다음부터는
    요약본에 "사회자가 X를 지시함" 한 줄로만 남습니다. 누적하면 라운드가
    갈수록 지시문이 헤더처럼 쌓여 컨텍스트 예산을 먹습니다.

    v1 은 전체 브로드캐스트만 지원합니다 — 참가자별 개별 지시는 컨텍스트를
    참가자마다 갈라놓아서, Judge 가 "왜 이 참가자만 이 얘길 하지"를 판단할
    근거를 잃습니다.

    투입은 슬라이스 4 입니다. 지금은 항상 비어 있습니다.
    """

    text: str
    round_no: int
    source: str = "human"


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
    #: 이번 라운드에 한해 적용되는 사람의 지시. 슬라이스 4 에서 채워집니다.
    directives: tuple[Directive, ...] = ()

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

        if self.directives:
            listed = "\n".join(f"  - {d.text}" for d in self.directives)
            # 쟁점 뒤, 차례 지시 앞. 쟁점보다 앞에 두면 배경으로 읽히고,
            # 차례 지시 뒤에 두면 무시됩니다.
            parts.append(
                f"[사회자 지시] 이번 라운드에 한해 아래를 반드시 반영하십시오.\n{listed}"
            )

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
    #: 토론 식별자. **한 번만 만들어 원장·저장·API 가 같은 값을 씁니다.**
    #: 엔진이 따로 만들면 debates 행과 llm_calls 행이 서로 다른 id 를 갖게 되어
    #: 원장이 토론에서 떨어져 나갑니다.
    debate_id: str
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
    #: **성공한** 발언들의 지연 합. wall_ms 와 비교하면 병렬 여부가 보입니다.
    #:
    #: 실패한 발언은 latency 가 0 이라 합계에 넣으면 지표가 거꾸로 읽힙니다 —
    #: 재시도로 3초를 태우고 죽은 참가자가 있으면 wall 은 크고 sum 은 작아져서,
    #: 병렬로 잘 돌았는데도 직렬처럼 보입니다. 그래서 성공분만 세고, 실패 건수는
    #: failed_count 로 따로 읽습니다. 둘을 같이 봐야 해석이 됩니다.
    sum_latency_ms: int
    #: ceil(참가자수 / 동시성). 세마포어 때문에 wall 이 max 보다 큰 이유.
    waves: int

    @property
    def ok_utterances(self) -> tuple[Utterance, ...]:
        return tuple(u for u in self.utterances if u.status == "ok")

    @property
    def ok_count(self) -> int:
        return len(self.ok_utterances)

    @property
    def failed_count(self) -> int:
        return len(self.utterances) - self.ok_count

    @property
    def max_latency_ms(self) -> int:
        """성공한 발언 중 가장 느린 것. 웨이브가 1이면 wall 의 하한입니다."""
        return max((u.latency_ms for u in self.ok_utterances), default=0)


@dataclass(slots=True)
class DebateState:
    """토론의 확정된 상태.

    **진행 중인 라운드를 담을 필드가 없습니다.** 이게 "같은 라운드 참가자는
    서로 못 본다"를 보장하는 방식입니다 — ContextBuilder 는 이 객체만 읽으므로,
    아직 커밋되지 않은 발언에 접근할 경로가 타입 상 존재하지 않습니다.
    """

    topic: str
    round_no: int = 1
    completed_rounds: list[RoundResult] = field(default_factory=list)
    issues: tuple[Issue, ...] = ()
    #: 직전 라운드를 뺀 그 이전 라운드들의 압축본.
    prior_digest: str = ""
    #: 이번 라운드에만 적용되는 지시. 라운드가 끝나면 비워집니다.
    directives: tuple[Directive, ...] = ()
    #: 요약본에 흔적으로 남길 지시 이력 (라운드번호, 요지).
    directive_trace: list[tuple[int, str]] = field(default_factory=list)

    @property
    def last_round(self) -> RoundResult | None:
        return self.completed_rounds[-1] if self.completed_rounds else None


DebateStatus = Literal["completed", "aborted_insufficient_participants"]


@dataclass(frozen=True, slots=True)
class DebateResult:
    debate_id: str
    topic: str
    participants: tuple[AgentSpec, ...]
    rounds: tuple[RoundResult, ...]
    status: DebateStatus = "completed"
    dropped: tuple[str, ...] = ()
    issues: tuple[Issue, ...] = ()
    #: 치명적이지 않은 문제들(쟁점 추출 실패 등). 토론은 계속되지만 조용히
    #: 넘어가면 안 되는 것들입니다.
    warnings: tuple[str, ...] = ()


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
class UtteranceStarted:
    """세마포어를 얻어 실제로 호출이 나간 시점.

    RoundStarted 의 active 목록과 구분됩니다 — 동시성 3 에 참가자 5명이면
    라운드가 시작돼도 2명은 아직 대기 중입니다. UI 가 "말하는 중"과
    "차례 기다리는 중"을 구분하려면 이 이벤트가 필요합니다.
    """

    agent_id: str
    label: str
    round_no: int


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
class IssuesExtracted:
    issues: tuple[Issue, ...]


@dataclass(frozen=True, slots=True)
class RoundSummarized:
    round_no: int
    digest: str


@dataclass(frozen=True, slots=True)
class RoundCompleted:
    result: RoundResult


@dataclass(frozen=True, slots=True)
class DebateCompleted:
    result: DebateResult


DebateEvent = Union[
    DebateStarted,
    RoundStarted,
    UtteranceStarted,
    UtteranceCompleted,
    AgentDropped,
    IssuesExtracted,
    RoundSummarized,
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
    #: 어느 라운드의 호출인지. 슬라이스 3 의 원장 조회가 라운드별 비용을
    #: 뽑으려면 필요합니다.
    round_no: int | None = None


class ConfigError(Exception):
    """설정이 잘못됨. LLM 을 한 번도 부르기 전에 던집니다."""
