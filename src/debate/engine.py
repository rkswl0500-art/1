"""DebateEngine.

라운드 내부는 병렬, 라운드 간은 순차입니다. 동시 실행 수는 세마포어로
제한하므로 참가자 5명 / 동시성 3 이면 한 라운드가 2 웨이브로 돕니다.

라운드 사이에 세 가지가 일어납니다:
  R1 직후      사회자가 쟁점 3~5개를 뽑습니다. 이후 라운드는 그 안에서만.
  매 라운드 전   InterventionGate 가 사람의 지시를 수집합니다(기본 no-op).
  R2 이후      직전의 직전 라운드를 요약본에 접습니다 — 컨텍스트 예산 관리.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Awaitable, Callable, Protocol, Sequence

from .agent import Agent, Anonymizer, Moderator
from .context import ContextBuilder
from .models import (
    AgentDropped,
    DebateCompleted,
    DebateConfig,
    DebateEvent,
    DebateResult,
    DebateStarted,
    DebateState,
    Directive,
    IssuesExtracted,
    RoundCompleted,
    RoundResult,
    RoundStarted,
    RoundSummarized,
    Utterance,
    UtteranceStarted,
    UtteranceCompleted,
)

EventSink = Callable[[DebateEvent], Awaitable[None]]


class InterventionGate(Protocol):
    """라운드 사이에 사람의 지시를 받는 자리.

    엔진은 이게 즉시 반환하는지 사람을 기다리는지 모릅니다. CLI 는 no-op 을,
    슬라이스 4 의 API 는 실제 대기를 꽂습니다. 지시는 **1회용**이라 수집된
    라운드에만 적용되고, 이후에는 요약본에 한 줄 흔적으로만 남습니다.
    """

    async def collect(self, round_no: int, state: DebateState) -> tuple[Directive, ...]: ...


class NoIntervention:
    """기본 게이트. 아무도 기다리지 않고 빈 튜플을 돌려줍니다."""

    async def collect(self, round_no: int, state: DebateState) -> tuple[Directive, ...]:
        return ()


async def _noop_sink(_: DebateEvent) -> None:
    return None


class DebateEngine:
    def __init__(
        self,
        agents: Sequence[Agent],
        builder: ContextBuilder,
        moderator: Moderator,
        anonymizer: Anonymizer,
        *,
        gate: InterventionGate | None = None,
        sink: EventSink | None = None,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("참가자는 최소 2명이어야 합니다")
        self._agents = list(agents)
        self._builder = builder
        self._moderator = moderator
        self._anon = anonymizer
        self._gate = gate or NoIntervention()
        self._sink = sink or _noop_sink

    async def run(self, cfg: DebateConfig) -> DebateResult:
        debate_id = cfg.debate_id
        specs = tuple(a.spec for a in self._agents)
        state = DebateState(topic=cfg.topic)
        warnings: list[str] = []
        dropped: list[str] = []
        alive = list(self._agents)

        await self._sink(DebateStarted(debate_id, cfg.topic, specs, cfg.rounds))

        for round_no in range(1, cfg.rounds + 1):
            state.round_no = round_no
            state.directives = await self._gate.collect(round_no, state)
            for directive in state.directives:
                state.directive_trace.append((round_no, directive.text))

            await self._sink(RoundStarted(round_no, tuple(a.id for a in alive)))
            result, failures = await self._run_round(round_no, alive, cfg, state)
            state.completed_rounds.append(result)

            for agent, err in failures:
                dropped.append(agent.id)
                await self._sink(
                    AgentDropped(agent.id, round_no, f"{type(err).__name__}: {err}")
                )
            if failures:
                failed = {a.id for a, _ in failures}
                alive = [a for a in alive if a.id not in failed]

            await self._sink(RoundCompleted(result))
            state.directives = ()          # 1회용: 다음 라운드로 넘기지 않습니다

            if len(alive) < 2:
                return await self._finish(
                    debate_id, cfg, specs, state, dropped, warnings,
                    status="aborted_insufficient_participants",
                )

            if round_no < cfg.rounds:
                await self._between_rounds(round_no, state, warnings)

        return await self._finish(debate_id, cfg, specs, state, dropped, warnings)

    async def _between_rounds(
        self, round_no: int, state: DebateState, warnings: list[str]
    ) -> None:
        """쟁점 추출(R1 직후)과 요약 접기(R2 이후).

        둘 다 실패해도 토론은 계속합니다 — 사회자가 한 번 삐끗했다고 진행된
        라운드를 버리는 건 과잉입니다. 대신 경고로 남겨서 조용히 넘어가지
        않게 합니다.
        """
        if round_no == 1:
            try:
                first = state.completed_rounds[0]
                anon = tuple(
                    self._anon.to_anon(u) for u in first.utterances if u.status == "ok"
                )
                state.issues = await self._moderator.extract_issues(state.topic, anon)
                await self._sink(IssuesExtracted(state.issues))
            except Exception as e:                      # noqa: BLE001
                warnings.append(
                    f"쟁점 추출 실패 ({type(e).__name__}: {e}). "
                    "이후 라운드가 쟁점 제약 없이 진행됩니다."
                )
            return

        # R_n 을 마쳤으면 R_(n-1) 을 요약본에 접습니다. 그래야 다음 라운드가
        # 직전 전문(R_n) + 그 이전 요약(R_1..R_(n-1)) 형태가 됩니다.
        target = state.completed_rounds[round_no - 2]
        try:
            state.prior_digest = await self._moderator.summarize(
                state.prior_digest, target, self._anon.to_anon,
                directive_trace=tuple(state.directive_trace),
            )
            await self._sink(RoundSummarized(target.round_no, state.prior_digest))
        except Exception as e:                          # noqa: BLE001
            warnings.append(
                f"R{target.round_no} 요약 실패 ({type(e).__name__}: {e}). "
                "이전 요약을 그대로 유지합니다."
            )

    async def _finish(
        self, debate_id, cfg, specs, state, dropped, warnings, *, status="completed",
    ) -> DebateResult:
        out = DebateResult(
            debate_id=debate_id,
            topic=cfg.topic,
            participants=specs,
            rounds=tuple(state.completed_rounds),
            status=status,
            dropped=tuple(dropped),
            issues=state.issues,
            warnings=tuple(warnings),
        )
        await self._sink(DebateCompleted(out))
        return out

    async def _run_round(
        self, round_no: int, agents: Sequence[Agent], cfg: DebateConfig,
        state: DebateState,
    ) -> tuple[RoundResult, list[tuple[Agent, BaseException]]]:
        sem = asyncio.Semaphore(max(1, cfg.max_concurrency))

        async def one(agent: Agent) -> Utterance:
            # 팩은 세마포어 밖에서 만듭니다 — state 는 이 시점에 이미 확정본이고,
            # 대기 중에 바뀌지 않습니다.
            ctx = self._builder.build_for(agent.spec, state)
            async with sem:
                # 세마포어를 얻은 뒤에 방출합니다. 그래야 "말하는 중"과
                # "차례를 기다리는 중"이 화면에서 갈립니다.
                await self._sink(UtteranceStarted(agent.id, agent.label, round_no))
                utterance = await asyncio.wait_for(agent.speak(ctx), cfg.round_timeout_s)
            # gather 가 끝난 뒤가 아니라 **완료 즉시** 방출합니다. 뒤로 미루면
            # 이벤트가 참가자 순서대로 한꺼번에 나가고, 지연 편차가 큰 라운드에서
            # 진행 표시가 전혀 진행 표시 구실을 못 합니다.
            await self._sink(UtteranceCompleted(utterance, agent.label))
            return utterance

        started = time.perf_counter()
        # return_exceptions=True 가 요구사항 7 의 핵심입니다. 한 명이 터져도
        # 나머지 코루틴은 끝까지 돌고, 실패는 값으로 돌아옵니다.
        settled = await asyncio.gather(*(one(a) for a in agents), return_exceptions=True)
        wall_ms = int((time.perf_counter() - started) * 1000)

        utterances: list[Utterance] = []
        failures: list[tuple[Agent, BaseException]] = []
        for agent, outcome in zip(agents, settled):
            if isinstance(outcome, BaseException):
                failures.append((agent, outcome))
                utterances.append(Agent.failed(agent.spec, round_no, outcome))
            else:
                utterances.append(outcome)  # 이벤트는 one() 에서 이미 방출됨

        result = RoundResult(
            round_no=round_no,
            utterances=tuple(utterances),
            wall_ms=wall_ms,
            # 성공분만 합산합니다. 실패한 발언(latency 0)을 섞으면 wall > sum 이
            # 되어 병렬성 지표가 거꾸로 읽힙니다.
            sum_latency_ms=sum(u.latency_ms for u in utterances if u.status == "ok"),
            waves=math.ceil(len(agents) / max(1, cfg.max_concurrency)),
        )
        return result, failures
