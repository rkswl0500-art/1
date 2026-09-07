"""DebateEngine.

라운드 내부는 병렬, 라운드 간은 순차입니다. 동시 실행 수는 세마포어로
제한하므로 참가자 5명 / 동시성 3 이면 한 라운드가 2 웨이브로 돕니다 —
벽시계 시간이 max(지연)이 아니라 대략 웨이브 수 × max(지연)이 되는 이유입니다.

슬라이스 1 범위: 1라운드. 다라운드는 ContextBuilder(슬라이스 2)가 있어야
의미가 있으므로 rounds > 1 이면 명시적으로 거부합니다 — 조용히 빈 컨텍스트로
2라운드를 돌려 쓰레기를 만드느니 죽는 게 낫습니다.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from typing import Awaitable, Callable, Sequence

from .agent import Agent, build_header
from .models import (
    AgentDropped,
    ContextPack,
    DebateCompleted,
    DebateConfig,
    DebateEvent,
    DebateResult,
    DebateStarted,
    RoundCompleted,
    RoundResult,
    RoundStarted,
    Utterance,
    UtteranceCompleted,
)

EventSink = Callable[[DebateEvent], Awaitable[None]]


async def _noop_sink(_: DebateEvent) -> None:
    return None


class DebateEngine:
    def __init__(self, agents: Sequence[Agent], *, sink: EventSink | None = None) -> None:
        if len(agents) < 2:
            raise ValueError("참가자는 최소 2명이어야 합니다")
        self._agents = list(agents)
        self._sink = sink or _noop_sink

    async def run(self, cfg: DebateConfig) -> DebateResult:
        if cfg.rounds != 1:
            raise NotImplementedError(
                f"슬라이스 1 은 1라운드만 지원합니다 (요청: {cfg.rounds}). "
                "다라운드는 ContextBuilder 가 붙는 슬라이스 2 범위입니다."
            )

        debate_id = f"d_{uuid.uuid4().hex[:6]}"
        specs = tuple(a.spec for a in self._agents)
        await self._sink(DebateStarted(debate_id, cfg.topic, specs, cfg.rounds))

        rounds: list[RoundResult] = []
        dropped: list[str] = []
        alive = list(self._agents)

        for round_no in range(1, cfg.rounds + 1):
            await self._sink(RoundStarted(round_no, tuple(a.id for a in alive)))
            result, failures = await self._run_round(round_no, alive, cfg)
            rounds.append(result)

            for agent, err in failures:
                dropped.append(agent.id)
                await self._sink(
                    AgentDropped(agent.id, round_no, f"{type(err).__name__}: {err}")
                )
            if failures:
                failed_ids = {a.id for a, _ in failures}
                alive = [a for a in alive if a.id not in failed_ids]

            await self._sink(RoundCompleted(result))

            if len(alive) < 2:
                out = DebateResult(
                    debate_id, cfg.topic, specs, tuple(rounds),
                    status="aborted_insufficient_participants", dropped=tuple(dropped),
                )
                await self._sink(DebateCompleted(out))
                return out

        out = DebateResult(
            debate_id, cfg.topic, specs, tuple(rounds), dropped=tuple(dropped)
        )
        await self._sink(DebateCompleted(out))
        return out

    async def _run_round(
        self, round_no: int, agents: Sequence[Agent], cfg: DebateConfig
    ) -> tuple[RoundResult, list[tuple[Agent, BaseException]]]:
        sem = asyncio.Semaphore(max(1, cfg.max_concurrency))

        async def one(agent: Agent) -> Utterance:
            ctx = self._context_for(agent, round_no, cfg)
            async with sem:
                return await asyncio.wait_for(agent.speak(ctx), cfg.round_timeout_s)

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
                utterances.append(outcome)
                await self._sink(UtteranceCompleted(outcome, agent.label))

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

    def _context_for(self, agent: Agent, round_no: int, cfg: DebateConfig) -> ContextPack:
        """슬라이스 1: 헤더만. 진행 중인 라운드는 구조상 참조 불가입니다.

        슬라이스 2 에서 이 자리를 ContextBuilder.build_for() 가 대체하며,
        확정된 이전 라운드(state.completed_rounds)만 읽는 불변식을 유지합니다.
        """
        return ContextPack(header=build_header(agent.spec, cfg.topic), round_no=round_no)
