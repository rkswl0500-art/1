"""토론 세션 — 조립, 이벤트 버퍼, 사람 개입 게이트.

api.py 를 얇게 유지하려고 HTTP 와 무관한 부분을 여기 모았습니다. 덕분에
세션 동작은 HTTP 없이 테스트됩니다.

**이벤트 버퍼가 있는 이유**: 브라우저를 닫았다 다시 열면 그 사이 이벤트는
이미 지나갔습니다. 버퍼가 없으면 재접속한 화면이 빈 채로 남아 토론이 죽은
것처럼 보입니다. 실제로는 서버에서 계속 돌고 있는데도요.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .agent import Agent, Anonymizer, Moderator
from .config import PricingTable, Settings
from .context import ContextBuilder
from .cost import CostEstimate, CostMeter, Estimator
from .engine import DebateEngine
from .judge import Judge, Verdict, family_note
from .models import (
    AgentDropped, AgentSpec, DebateCompleted, DebateConfig, DebateResult,
    DebateStarted, DebateState, Directive, IssuesExtracted, RoundCompleted,
    RoundStarted, RoundSummarized, UtteranceCompleted, UtteranceStarted,
)
from .provider import ProviderPool
from .storage import SqliteStore

#: 개입 대기 기본 상한. 라운드를 읽고 한 줄 치기에는 넉넉하고, 잊고 닫은 탭이
#: 토론을 붙잡아 두기에는 짧습니다. 0 이면 대기 자체를 하지 않습니다.
DEFAULT_GATE_TIMEOUT_S = 120.0

#: _safe() 가 실패를 알리는 표식.
_FAILED = object()


def event_to_dict(event: Any) -> dict:
    """엔진 이벤트를 SSE 로 보낼 수 있는 형태로."""
    if isinstance(event, DebateStarted):
        return {"type": "debate_started", "debate_id": event.debate_id,
                "topic": event.topic, "rounds": event.rounds,
                "participants": [{"id": s.id, "label": s.label, "model": s.model,
                                  "provider": s.provider, "stance": s.stance}
                                 for s in event.participants]}
    if isinstance(event, RoundStarted):
        return {"type": "round_started", "round_no": event.round_no,
                "active": list(event.active)}
    if isinstance(event, UtteranceStarted):
        return {"type": "utterance_started", "agent_id": event.agent_id,
                "label": event.label, "round_no": event.round_no}
    if isinstance(event, UtteranceCompleted):
        u = event.utterance
        return {"type": "utterance", "agent_id": u.agent_id, "label": event.label,
                "round_no": u.round_no, "content": u.content,
                "latency_ms": u.latency_ms, "in_tok": u.usage.prompt_tokens,
                "out_tok": u.usage.completion_tokens, "cost_usd": str(u.cost_usd),
                "finish_reason": u.finish_reason, "truncated": u.truncated}
    if isinstance(event, AgentDropped):
        return {"type": "agent_dropped", "agent_id": event.agent_id,
                "round_no": event.round_no, "reason": event.reason}
    if isinstance(event, IssuesExtracted):
        return {"type": "issues", "issues": [{"id": i.id, "title": i.title}
                                             for i in event.issues]}
    if isinstance(event, RoundSummarized):
        return {"type": "summarized", "round_no": event.round_no,
                "chars": len(event.digest)}
    if isinstance(event, RoundCompleted):
        r = event.result
        return {"type": "round_completed", "round_no": r.round_no,
                "wall_ms": r.wall_ms, "sum_latency_ms": r.sum_latency_ms,
                "max_latency_ms": r.max_latency_ms, "waves": r.waves,
                "ok": r.ok_count, "failed": r.failed_count}
    if isinstance(event, DebateCompleted):
        r = event.result
        return {"type": "debate_completed", "status": r.status,
                "dropped": list(r.dropped), "warnings": list(r.warnings)}
    return {"type": "unknown"}


@dataclass(frozen=True, slots=True)
class AwaitingDirective:
    """라운드 사이에 사람을 기다리는 중이라는 신호."""

    round_no: int
    timeout_s: float


class HttpGate:
    """사람의 지시를 기다리는 게이트.

    타임아웃은 **실패가 아니라 진행**입니다. 아무도 안 들어오면 지시 없이
    다음 라운드를 시작합니다 — 사람이 자리를 비웠다고 토론을 버리는 건 과잉이고,
    그 판단을 여기서 조용히 내리는 편이 낫습니다.
    """

    def __init__(self, timeout_s: float, emit) -> None:
        self.timeout_s = timeout_s
        self._emit = emit
        self._pending: list[str] = []
        self._release = asyncio.Event()
        self.waiting_round: int | None = None
        #: 실제로 사람을 기다린 시간의 합. 브라우저 닫힘 위험을 재는 근거.
        self.total_waited_s = 0.0

    async def collect(self, round_no: int, state: DebateState) -> tuple[Directive, ...]:
        if self.timeout_s <= 0 or round_no == 1:
            return ()                      # R1 전에는 볼 게 없어서 묻지 않습니다

        self.waiting_round = round_no
        self._release.clear()
        await self._emit(AwaitingDirective(round_no, self.timeout_s))

        started = time.perf_counter()
        try:
            await asyncio.wait_for(self._release.wait(), self.timeout_s)
        except asyncio.TimeoutError:
            pass                            # 지시 없이 진행
        finally:
            self.total_waited_s += time.perf_counter() - started
            self.waiting_round = None

        texts, self._pending = self._pending, []
        return tuple(Directive(text=t, round_no=round_no) for t in texts)

    def submit(self, text: str) -> bool:
        """지시를 넣고 즉시 라운드를 재개합니다."""
        if self.waiting_round is None:
            return False
        if text.strip():
            self._pending.append(text.strip())
        self._release.set()
        return True

    def proceed(self) -> bool:
        """지시 없이 바로 진행. 기다림을 사람이 끊습니다."""
        if self.waiting_round is None:
            return False
        self._release.set()
        return True


@dataclass
class DebateSession:
    debate_id: str
    topic: str
    specs: list[AgentSpec]
    config: DebateConfig
    estimate: CostEstimate
    meter: CostMeter
    pool: ProviderPool
    pricing: PricingTable
    settings: Settings
    judge_spec: tuple[str, str] | None
    gate: HttpGate
    status: str = "created"
    events: list[dict] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    result: DebateResult | None = None
    verdict: Verdict | None = None
    task: asyncio.Task | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    #: 판정 실패 사유. 토론은 성공했는데 심판만 실패한 경우를 구분합니다.
    judge_error: str | None = None

    # ── 이벤트 팬아웃 ────────────────────────────────────────────────────
    async def emit(self, event: Any) -> None:
        if isinstance(event, AwaitingDirective):
            payload = {"type": "awaiting_directive", "round_no": event.round_no,
                       "timeout_s": event.timeout_s}
        else:
            payload = event_to_dict(event)
        payload["seq"] = len(self.events)
        payload["running"] = self.running_totals()
        self.events.append(payload)
        for queue in list(self.subscribers):
            queue.put_nowait(payload)

    def running_totals(self) -> dict:
        """진행 중 누적. 무료 티어에서는 금액이 0 이라 토큰과 호출 수가 주 지표입니다."""
        rep = self.meter.report()
        return {"calls": rep.calls, "in_tok": rep.prompt_tokens,
                "out_tok": rep.completion_tokens, "total_tok": rep.total_tokens,
                "cost_usd": str(rep.total_usd.quantize(Decimal("0.000001")))}

    def subscribe(self, after_seq: int | None = None) -> asyncio.Queue:
        """구독 시작 시 밀린 이벤트를 먼저 밀어 넣습니다.

        after_seq 는 브라우저가 보낸 Last-Event-ID 입니다. 자동 재연결이면
        브라우저가 마지막으로 받은 seq 를 보내므로 그 **이후만** 재생합니다.
        새 탭이나 새로고침이면 헤더가 없으므로 전체를 재생합니다 — DOM 이
        비어 있어서 전부 필요하니까요. EventSource 가 자동 재연결일 때만
        헤더를 붙이므로 이 구분은 공짜로 얻어집니다.
        """
        queue: asyncio.Queue = asyncio.Queue()
        for payload in self.events:
            if after_seq is None or payload.get("seq", -1) > after_seq:
                queue.put_nowait(payload)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)

    # ── 실행 ────────────────────────────────────────────────────────────
    async def run(self) -> None:
        self.status, self.started_at = "running", time.perf_counter()
        try:
            anonymizer = Anonymizer(self.specs)
            builder = ContextBuilder(anonymizer)
            moderator = Moderator(self.pool.get(self.specs[0].provider),
                                  self.specs[0].model,
                                  timeout_s=self.settings.request_timeout_s)
            agents = [
                Agent(s, self.pool.get(s.provider), self.pricing,
                      max_tokens=self.settings.max_output_tokens,
                      timeout_s=self.settings.request_timeout_s)
                for s in self.specs
            ]
            engine = DebateEngine(agents, builder, moderator, anonymizer,
                                  gate=self.gate, sink=self.emit)
            self.result = await engine.run(self.config)

            if self.judge_spec and self._has_content():
                await self._judge(anonymizer)

            self.status = "completed"
        except Exception as e:                            # noqa: BLE001
            self.status, self.error = "failed", f"{type(e).__name__}: {e}"
            await self._safe(self.emit_raw({"type": "error", "message": self.error}))
        finally:
            await self.finalize()

    async def finalize(self) -> None:
        """종료 절차. **어떤 단계가 실패해도 finished 는 반드시 나갑니다.**

        스트림이 안 닫히면 EventSource 가 영원히 재연결하고, 사용자는 토론이
        멈춘 건지 끝난 건지 알 수 없습니다. 그래서 각 단계를 따로 감쌉니다.
        """
        self.finished_at = time.perf_counter()
        if await self._safe(self._persist()) is _FAILED:
            await self._safe(self.emit_raw({
                "type": "storage_failed",
                "message": "기록 저장에 실패했습니다. 토론 결과는 화면에만 있습니다.",
            }))
        await self._safe(self.emit_raw({
            "type": "finished",
            # 토론의 실제 결과를 씁니다. self.status 는 "run() 이 예외 없이
            # 끝났다"는 뜻이라, 참가자가 전원 죽어 중단된 토론도 completed 로
            # 보입니다 — 화면에 그대로 나가면 거짓말이 됩니다.
            "status": self.result.status if self.result else self.status,
            "run_status": self.status,
            "judge_error": self.judge_error,
            "waited_s": round(self.gate.total_waited_s, 1),
        }))
        await self._safe(self.pool.aclose())

    def _has_content(self) -> bool:
        """성공한 발언이 하나도 없으면 채점할 게 없습니다."""
        return any(u.status == "ok"
                   for r in (self.result.rounds if self.result else ())
                   for u in r.utterances)

    async def _judge(self, anonymizer) -> None:
        """판정. 실패해도 토론 전체를 죽이지 않고 화면에 사유를 띄웁니다.

        전체 시한을 따로 겁니다. 재시도 예산(retry_attempts × request_timeout_s)이
        기본값에서 6분이라, 심판이 무응답이면 그 6분 동안 아무 이벤트도 안 나가고
        화면이 멈춘 것처럼 보입니다.
        """
        assert self.result is not None and self.judge_spec is not None
        transcript = tuple(
            anonymizer.to_anon(u)
            for r in self.result.rounds for u in r.utterances if u.status == "ok"
        )
        judge = Judge(self.pool.get(self.judge_spec[0]), self.judge_spec[1],
                      max_tokens=self.settings.judge_max_tokens or None,
                      timeout_s=self.settings.request_timeout_s)
        # debate_completed 이후 판정까지 몇 분이 비면 화면이 멈춘 것과
        # 구분되지 않습니다. 시작을 알리고 UI 가 경과 시간을 돌립니다.
        await self.emit_raw({
            "type": "judging_started",
            "model": self.judge_spec[1],
            "timeout_s": self.settings.judge_timeout_s,
            "issues": len(self.result.issues),
        })
        try:
            self.verdict = await asyncio.wait_for(
                judge.evaluate(self.topic, self.result.issues, transcript,
                               labels=[s.label for s in self.specs]),
                self.settings.judge_timeout_s,
            )
            if self.verdict is not None and self.verdict.status == "unparsed":
                # 잘린 것과 형식이 깨진 것은 대처가 다릅니다. 하나로 묶어
                # "파싱 실패"라고만 하면 예산 문제를 프롬프트 문제로 오해합니다.
                kind = self.verdict.failure_kind
                await self._verdict_failed(
                    "심판 출력이 길이 제한에 걸려 중간에 잘렸습니다 "
                    f"(finish_reason=length). DEBATE_JUDGE_MAX_TOKENS 를 올리거나 "
                    "쟁점 수를 줄이십시오."
                    if kind == "truncated" else
                    "심판이 올바른 JSON 을 반환하지 않았습니다 (형식 오류). "
                    "원문은 verdicts.raw 에 저장했습니다.",
                    kind=kind)
            else:
                await self.emit_verdict()
        except asyncio.TimeoutError:
            await self._verdict_failed(
                f"심판이 {self.settings.judge_timeout_s:.0f}초 안에 응답하지 "
                "않았습니다 (DEBATE_JUDGE_TIMEOUT_S)")
        except Exception as e:                            # noqa: BLE001
            await self._verdict_failed(f"{type(e).__name__}: {e}")

    async def _verdict_failed(self, reason: str, kind: str = "error") -> None:
        self.judge_error = reason
        await self.emit_raw({"type": "verdict_failed", "message": reason,
                             "kind": kind})

    @staticmethod
    async def _safe(coro):
        """한 단계의 실패가 종료 절차 전체를 막지 못하게 합니다."""
        try:
            return await coro
        except Exception:                                 # noqa: BLE001
            return _FAILED

    async def emit_raw(self, payload: dict) -> None:
        payload["seq"] = len(self.events)
        payload["running"] = self.running_totals()
        self.events.append(payload)
        for queue in list(self.subscribers):
            queue.put_nowait(payload)

    async def emit_verdict(self) -> None:
        v = self.verdict
        assert v is not None
        await self.emit_raw({
            "type": "verdict", "status": v.status, "judge_model": v.judge_model,
            "winner": v.winner, "margin": v.margin, "conclusion": v.conclusion,
            "dissent": v.dissent, "totals": v.totals(),
            "rubric": {k: dict(s) for k, s in v.rubric.items()},
            "per_issue": [{"issue_id": s.issue_id, "scores": dict(s.scores),
                           "reasoning": s.reasoning} for s in v.per_issue],
            "truncated": v.truncated, "prompt_tokens": v.prompt_tokens,
        })

    async def _persist(self) -> None:
        if self.result is None:
            return
        note = (family_note(self.judge_spec[1], [s.model for s in self.specs])
                if self.judge_spec else None)
        store = SqliteStore(self.settings.db_path)
        try:
            store.save(self.result, self.meter, self.verdict, note)
        finally:
            store.close()


def new_debate_id() -> str:
    return f"d_{uuid.uuid4().hex[:6]}"
