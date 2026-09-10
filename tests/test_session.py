"""세션 계층: 개입 게이트와 이벤트 버퍼.

HTTP 없이 테스트합니다 — 게이트 동작은 라우팅과 무관합니다.
"""

from __future__ import annotations

import asyncio

import pytest

from debate.models import DebateState
from debate.session import DEFAULT_GATE_TIMEOUT_S, AwaitingDirective, HttpGate


async def _emit_nothing(_event) -> None:
    return None


def _gate(timeout: float, emit=_emit_nothing) -> HttpGate:
    return HttpGate(timeout, emit)


# ── 게이트 ───────────────────────────────────────────────────────────────────


async def test_gate_disabled_never_waits():
    gate = _gate(0.0)
    assert await gate.collect(2, DebateState(topic="t")) == ()
    assert gate.total_waited_s == 0.0


async def test_gate_does_not_ask_before_round_one():
    """R1 전에는 보여줄 게 없습니다. 물어봐야 답할 근거가 없습니다."""
    gate = _gate(30.0)
    assert await gate.collect(1, DebateState(topic="t")) == ()
    assert gate.waiting_round is None


async def test_timeout_proceeds_without_a_directive():
    """타임아웃은 실패가 아니라 진행입니다. 사람이 자리를 비웠다고 토론을
    버리는 건 과잉입니다."""
    gate = _gate(0.15)
    assert await gate.collect(2, DebateState(topic="t")) == ()
    assert gate.total_waited_s >= 0.15
    assert gate.waiting_round is None


async def test_submitted_directive_reaches_the_round_and_releases_it():
    gate = _gate(10.0)
    task = asyncio.create_task(gate.collect(2, DebateState(topic="t")))
    await asyncio.sleep(0.05)

    assert gate.submit("참가자 B의 통계 출처를 캐물어라") is True
    directives = await asyncio.wait_for(task, 1.0)

    assert len(directives) == 1
    assert directives[0].round_no == 2
    assert directives[0].source == "human"
    assert gate.total_waited_s < 1.0        # 타임아웃까지 안 기다림


async def test_proceed_skips_the_wait_without_a_directive():
    gate = _gate(10.0)
    task = asyncio.create_task(gate.collect(2, DebateState(topic="t")))
    await asyncio.sleep(0.05)

    assert gate.proceed() is True
    assert await asyncio.wait_for(task, 1.0) == ()


async def test_submitting_outside_a_wait_is_rejected():
    gate = _gate(10.0)
    assert gate.submit("아무거나") is False
    assert gate.proceed() is False


async def test_blank_directive_releases_but_adds_nothing():
    gate = _gate(10.0)
    task = asyncio.create_task(gate.collect(2, DebateState(topic="t")))
    await asyncio.sleep(0.05)
    gate.submit("   ")
    assert await asyncio.wait_for(task, 1.0) == ()


async def test_directives_do_not_leak_into_the_next_round():
    """1회용. 다음 라운드가 지난 지시를 다시 받으면 안 됩니다."""
    gate = _gate(0.1)
    task = asyncio.create_task(gate.collect(2, DebateState(topic="t")))
    await asyncio.sleep(0.02)
    gate.submit("R2 전용")
    assert len(await asyncio.wait_for(task, 1.0)) == 1

    assert await gate.collect(3, DebateState(topic="t")) == ()   # 타임아웃, 잔여 없음


async def test_awaiting_event_carries_the_deadline():
    seen: list = []

    async def emit(event):
        seen.append(event)

    gate = _gate(0.1, emit)
    await gate.collect(2, DebateState(topic="t"))

    assert len(seen) == 1
    assert isinstance(seen[0], AwaitingDirective)
    assert seen[0].round_no == 2 and seen[0].timeout_s == 0.1


def test_default_timeout_is_bounded():
    """잊고 닫은 탭이 토론을 무한정 붙잡으면 안 됩니다."""
    assert 30 <= DEFAULT_GATE_TIMEOUT_S <= 300


# ── 이벤트 버퍼 (재접속 복구) ────────────────────────────────────────────────


async def test_late_subscriber_receives_the_backlog():
    """브라우저를 닫았다 다시 열면 그 사이 이벤트는 지나갔습니다. 버퍼가 없으면
    재접속한 화면이 빈 채로 남아 토론이 죽은 것처럼 보입니다."""
    from debate.session import DebateSession

    session = DebateSession.__new__(DebateSession)
    session.events = [{"type": "a"}, {"type": "b"}, {"type": "c"}]
    session.subscribers = set()

    queue = session.subscribe()
    assert queue.qsize() == 3
    assert [queue.get_nowait()["type"] for _ in range(3)] == ["a", "b", "c"]

    session.unsubscribe(queue)
    assert session.subscribers == set()
