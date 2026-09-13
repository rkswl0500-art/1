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


# ── 재연결 시 중복 재생 (실측으로 드러난 버그) ──────────────────────────────


def _session_with(events: list[dict]):
    from debate.session import DebateSession

    s = DebateSession.__new__(DebateSession)
    s.events = events
    s.subscribers = set()
    return s


def test_reconnect_cursor_sends_only_later_events():
    """브라우저는 자동 재연결 시 마지막 id 를 Last-Event-ID 로 돌려줍니다.
    그 이후만 보내지 않으면 재연결마다 전체 버퍼가 다시 흘러가 화면이
    중복 누적됩니다 — 실제로 20회 재연결에 발언이 36개까지 불어났습니다."""
    session = _session_with([{"type": f"e{i}", "seq": i} for i in range(6)])

    queue = session.subscribe(after_seq=2)
    got = [queue.get_nowait()["seq"] for _ in range(queue.qsize())]

    assert got == [3, 4, 5]


def test_fresh_subscriber_without_cursor_gets_everything():
    """새 탭·새로고침은 DOM 이 비어 있으므로 전체가 필요합니다. EventSource 는
    자동 재연결일 때만 헤더를 붙이므로 이 구분은 공짜로 얻어집니다."""
    session = _session_with([{"type": f"e{i}", "seq": i} for i in range(4)])

    assert session.subscribe().qsize() == 4
    assert session.subscribe(after_seq=None).qsize() == 4


def test_cursor_at_the_end_yields_nothing():
    """토론이 끝난 뒤 다시 붙어도 보낼 게 없어야 합니다."""
    session = _session_with([{"type": f"e{i}", "seq": i} for i in range(4)])
    assert session.subscribe(after_seq=3).qsize() == 0


def test_cursor_beyond_the_buffer_is_not_an_error():
    session = _session_with([{"type": "e0", "seq": 0}])
    assert session.subscribe(after_seq=999).qsize() == 0


def test_every_emitted_event_carries_a_monotonic_seq():
    """클라이언트 중복 제거가 seq 에 기대므로 빠지거나 뒤섞이면 안 됩니다."""
    import asyncio

    from debate.config import PricingTable
    from debate.cost import CostMeter
    from debate.session import DebateSession

    s = DebateSession.__new__(DebateSession)
    s.events, s.subscribers = [], set()
    s.meter = CostMeter(PricingTable({}), "d")

    async def run():
        for i in range(5):
            await s.emit_raw({"type": f"e{i}"})

    asyncio.run(run())
    assert [e["seq"] for e in s.events] == [0, 1, 2, 3, 4]


# ── 종료 이벤트는 어떤 실패에도 나가야 한다 (라이브에서 드러남) ─────────────


def _runnable_session(**kw):
    """run() 의 종료 절차만 떼어 검사하기 위한 최소 세션."""
    from debate.config import PricingTable, Settings
    from debate.cost import CostMeter
    from debate.session import DebateSession

    s = DebateSession.__new__(DebateSession)
    s.events, s.subscribers = [], set()
    s.meter = CostMeter(PricingTable({}), "d")
    s.settings = Settings(_env_file=None)
    s.gate = HttpGate(0.0, _emit_nothing)
    s.status, s.error, s.judge_error = "running", None, None
    s.result = s.verdict = s.judge_spec = None
    s.pool = type("P", (), {"aclose": staticmethod(lambda: asyncio.sleep(0))})()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


async def _finish_like_run(session, persist):
    """실제 종료 절차를 호출합니다. 테스트가 finally 블록을 베껴 쓰면 코드가
    바뀌었을 때 테스트만 통과하는 상태가 됩니다."""
    session._persist = persist
    await session.finalize()


async def test_finished_is_emitted_even_when_storage_fails():
    """저장이 터지면 finished 가 통째로 사라져 스트림이 영원히 안 닫혔습니다.
    EventSource 는 그걸 끊김으로 보고 계속 재연결합니다."""
    session = _runnable_session()

    async def boom():
        raise OSError("disk full")

    await _finish_like_run(session, boom)
    types = [e["type"] for e in session.events]

    assert "storage_failed" in types      # 조용히 넘어가지 않음
    assert types[-1] == "finished"        # 그래도 스트림은 닫힘


async def test_finished_is_emitted_after_a_clean_save():
    session = _runnable_session()

    async def fine():
        return None

    await _finish_like_run(session, fine)
    assert [e["type"] for e in session.events] == ["finished"]


async def test_safe_swallows_only_the_failing_stage():
    from debate.session import _FAILED

    session = _runnable_session()

    async def boom():
        raise RuntimeError("x")

    async def ok():
        return "값"

    assert await session._safe(boom()) is _FAILED
    assert await session._safe(ok()) == "값"


async def test_judge_failure_is_reported_not_swallowed():
    """판정 실패는 화면에 사유가 떠야 합니다. 토론 자체는 살아 있습니다."""
    session = _runnable_session()
    await session._verdict_failed("RetryableError: 3회 시도 모두 실패")

    assert session.judge_error is not None
    assert session.events[-1]["type"] == "verdict_failed"
    assert "3회 시도" in session.events[-1]["message"]


def test_judge_has_its_own_deadline_shorter_than_the_retry_budget():
    """재시도 예산이 기본 6분이라, 심판 무응답 시 그동안 이벤트가 안 나갑니다."""
    from debate.config import Settings

    s = Settings(_env_file=None)
    assert s.judge_timeout_s < s.retry_budget_s
    assert 60 <= s.judge_timeout_s <= 300


async def test_finished_reports_the_debate_outcome_not_just_that_run_returned():
    """세션 status 는 'run() 이 예외 없이 끝났다'는 뜻입니다. 참가자가 전원 죽어
    중단된 토론도 그 값으로는 completed 로 보입니다."""
    from debate.models import AgentSpec, DebateResult

    session = _runnable_session()
    session.status = "completed"
    session.result = DebateResult(
        debate_id="d", topic="t",
        participants=(AgentSpec("p1", "참가자 A", "fake", "m", "x"),),
        rounds=(), status="aborted_insufficient_participants")
    session.judge_error = "RetryableError: 심판 무응답"

    async def fine():
        return None

    await _finish_like_run(session, fine)
    last = session.events[-1]

    assert last["type"] == "finished"
    assert last["status"] == "aborted_insufficient_participants"
    assert last["run_status"] == "completed"      # 둘을 구분해서 싣습니다
    assert last["judge_error"] == "RetryableError: 심판 무응답"


async def test_judge_is_skipped_when_no_participant_succeeded():
    """빈 기록을 채점시킬 이유가 없습니다. 참가자가 전원 죽으면 심판도 건너뜁니다."""
    from debate.models import AgentSpec, DebateResult, RoundResult, Usage, Utterance
    from decimal import Decimal

    session = _runnable_session()
    failed = Utterance("p1", 1, "", Usage(0, 0), 0, Decimal(0),
                       status="failed", error="boom")
    session.result = DebateResult(
        debate_id="d", topic="t",
        participants=(AgentSpec("p1", "참가자 A", "fake", "m", "x"),),
        rounds=(RoundResult(1, (failed,), 0, 0, 1),))

    assert session._has_content() is False


async def test_judge_runs_when_at_least_one_utterance_succeeded():
    from debate.models import AgentSpec, DebateResult, RoundResult, Usage, Utterance
    from decimal import Decimal

    session = _runnable_session()
    ok = Utterance("p1", 1, "발언", Usage(1, 1), 1, Decimal(0))
    session.result = DebateResult(
        debate_id="d", topic="t",
        participants=(AgentSpec("p1", "참가자 A", "fake", "m", "x"),),
        rounds=(RoundResult(1, (ok,), 0, 0, 1),))

    assert session._has_content() is True


def test_judge_timeout_accommodates_a_slow_thinking_model():
    """라이브에서 사고형 심판이 3~4분 걸렸습니다. 180초면 정상 판정을 끊습니다."""
    from debate.config import Settings

    s = Settings(_env_file=None)
    assert s.judge_timeout_s >= 240
