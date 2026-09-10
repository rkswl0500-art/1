"""FastAPI + SSE.

라우팅만 있습니다. 조립·이벤트 버퍼·개입 게이트는 session.py 에 있어서 HTTP
없이도 테스트됩니다.

**응답에 키·base_url 이 절대 나가지 않습니다.** 프론트는 /debates/* 만 부르고
프로바이더가 어디 있는지 모릅니다.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import DEFAULT_PERSONA_FALLBACK
from .config import FAKE_PROVIDER, PricingTable, Settings, load_provider_slots
from .cost import CostMeter, Estimator
from .judge import family_note, vendor_family
from .models import AgentSpec, ConfigError
from .provider import FakeProvider, build_pool
from .session import (
    DEFAULT_GATE_TIMEOUT_S, DebateSession, HttpGate, new_debate_id,
)

app = FastAPI(title="AI Debate Room")
_SESSIONS: dict[str, DebateSession] = {}
_STATIC = Path(__file__).parent / "static"


class Participant(BaseModel):
    provider: str
    model: str
    persona: str = DEFAULT_PERSONA_FALLBACK
    stance: str | None = None


class CreateDebate(BaseModel):
    topic: str = Field(min_length=1)
    participants: list[Participant] = Field(min_length=2, max_length=5)
    rounds: int = Field(default=3, ge=1, le=10)
    judge: str | None = None                    # "provider/model"
    allow_judge_overlap: bool = False
    use_fake: bool = False
    #: fake 모드 전용. {모델: 지연ms}. 느린 참가자를 기다리는 화면을 재현하려면
    #: 실제 지연 편차가 필요합니다 (관측: 54.3s vs 1.8s).
    fake_latency_ms: dict[str, int] | None = None
    #: 0 이면 라운드 사이에 사람을 기다리지 않습니다.
    gate_timeout_s: float = Field(default=0.0, ge=0, le=1800)


class DirectiveIn(BaseModel):
    text: str = ""


def _label(i: int) -> str:
    return f"참가자 {chr(65 + i)}"


def _make_fake(body: CreateDebate) -> FakeProvider:
    from .provider import FakeBehavior

    lat = body.fake_latency_ms or {}
    return FakeProvider({m: FakeBehavior(latency_ms=ms) for m, ms in lat.items()})


def _session(debate_id: str) -> DebateSession:
    try:
        return _SESSIONS[debate_id]
    except KeyError:
        raise HTTPException(404, f"debate {debate_id} 를 찾을 수 없습니다") from None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/models")
async def models(provider: str = FAKE_PROVIDER) -> dict:
    """모델 ID 목록. base_url·키는 응답에 넣지 않습니다."""
    if provider == FAKE_PROVIDER:
        return {"provider": provider, "models": await FakeProvider().list_models()}
    registry = load_provider_slots()
    if provider not in registry:
        raise HTTPException(400, f"프로바이더 {provider!r} 없음. "
                                 f"사용 가능: {', '.join(registry.names()) or '<없음>'}")
    from .provider import OpenAICompatProvider

    p = OpenAICompatProvider(registry.get(provider),
                             timeout_s=Settings().request_timeout_s)
    try:
        return {"provider": provider, "models": await p.list_models()}
    finally:
        await p.aclose()


@app.get("/providers")
async def providers() -> dict:
    """설정된 프로바이더 **이름만**. base_url 도 키도 내보내지 않습니다."""
    return {"providers": list(load_provider_slots().names()), "fake": FAKE_PROVIDER}


@app.post("/debates")
async def create(body: CreateDebate) -> dict:
    """생성 = 견적. **LLM 을 한 번도 부르지 않습니다.**

    시작과 나뉘어 있는 이유는 비용을 보고 결정할 수 있어야 하기 때문입니다.
    """
    settings = Settings()
    specs = [
        AgentSpec(id=f"p{i+1}", label=_label(i),
                  provider=FAKE_PROVIDER if body.use_fake else p.provider,
                  model=p.model, persona=p.persona, stance=p.stance)
        for i, p in enumerate(body.participants)
    ]

    judge_spec = None
    if body.judge:
        provider, _, model = body.judge.rpartition("/")
        provider = provider or specs[0].provider
        if not model:
            raise HTTPException(400, "judge 는 'provider/model' 형식이어야 합니다")
        if any(s.model == model for s in specs) and not body.allow_judge_overlap:
            raise HTTPException(400, (
                f"judge 모델 {model!r} 이 참가자 풀에 있습니다. LLM 은 자기 출력을 "
                "편애해서 점수가 오염됩니다. allow_judge_overlap 으로 명시 허용 가능."))
        judge_spec = (FAKE_PROVIDER if body.use_fake else provider, model)

    pricing = PricingTable.load(settings.pricing_path)
    debate_id = new_debate_id()
    meter = CostMeter(pricing, debate_id)

    pool_specs = list(specs)
    if judge_spec:
        pool_specs.append(AgentSpec("__judge__", "__judge__", judge_spec[0],
                                    judge_spec[1], ""))
    try:
        pool = build_pool(
            pool_specs, None if body.use_fake else load_provider_slots(),
            meter, settings,
            fake=_make_fake(body) if body.use_fake else None,
        )
    except ConfigError as e:
        raise HTTPException(400, str(e)) from None

    estimate = Estimator(pricing, settings.ko_tokens_per_char).estimate(
        participants=specs, rounds=body.rounds,
        judge_model=judge_spec[1] if judge_spec else None)

    from .models import DebateConfig

    session = DebateSession(
        debate_id=debate_id, topic=body.topic, specs=specs,
        config=DebateConfig(debate_id=debate_id, topic=body.topic,
                            participants=tuple(specs),
                            rounds=body.rounds,
                            max_concurrency=settings.max_concurrency,
                            round_timeout_s=settings.round_timeout_s),
        estimate=estimate, meter=meter, pool=pool, pricing=pricing,
        settings=settings, judge_spec=judge_spec,
        gate=HttpGate(body.gate_timeout_s, lambda e: _SESSIONS[debate_id].emit(e)),
    )
    _SESSIONS[debate_id] = session

    note = family_note(judge_spec[1], [s.model for s in specs]) if judge_spec else None
    return {
        "debate_id": debate_id,
        "estimate": {
            "calls": estimate.calls,
            "tokens_low": estimate.tokens_low, "tokens_high": estimate.tokens_high,
            "low_usd": str(estimate.low_usd), "high_usd": str(estimate.high_usd),
            "unpriced": list(estimate.unpriced_models),
            "ko_tokens_per_char": estimate.ko_tokens_per_char,
        },
        "participants": [{"id": s.id, "label": s.label, "model": s.model} for s in specs],
        "judge": {"model": judge_spec[1], "family": vendor_family(judge_spec[1]),
                  "shares_family_with_participants": note.shares_family,
                  "note": note.line()} if judge_spec and note else None,
        "gate_timeout_s": body.gate_timeout_s,
        "llm_calls_made": 0,
    }


@app.post("/debates/{debate_id}/start", status_code=202)
async def start(debate_id: str) -> dict:
    session = _session(debate_id)
    if session.task is not None:
        raise HTTPException(409, "이미 시작되었습니다")
    # 백그라운드 태스크입니다. 브라우저를 닫아도 서버에서 계속 돕니다.
    session.task = asyncio.create_task(session.run())
    return {"status": "started", "debate_id": debate_id}


@app.get("/debates/{debate_id}/stream")
async def stream(debate_id: str) -> StreamingResponse:
    """SSE. 접속 시 지금까지의 이벤트를 먼저 재생하고 이후 실시간으로 흘립니다."""
    session = _session(debate_id)

    async def gen():
        queue = session.subscribe()
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"      # 프록시가 끊지 않게
                    continue
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                if payload.get("type") == "finished":
                    break
        finally:
            session.unsubscribe(queue)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache",
                                      "x-accel-buffering": "no"})


@app.post("/debates/{debate_id}/directive")
async def directive(debate_id: str, body: DirectiveIn) -> dict:
    """라운드 사이에 지시를 넣습니다. 1회용이고 전체에게 브로드캐스트됩니다."""
    session = _session(debate_id)
    if not session.gate.submit(body.text):
        raise HTTPException(409, "지금은 지시를 받는 시점이 아닙니다")
    return {"accepted": True, "round_no": session.gate.waiting_round}


@app.post("/debates/{debate_id}/proceed")
async def proceed(debate_id: str) -> dict:
    """기다리지 않고 바로 다음 라운드로."""
    session = _session(debate_id)
    if not session.gate.proceed():
        raise HTTPException(409, "지금은 대기 중이 아닙니다")
    return {"accepted": True}


@app.get("/debates/{debate_id}")
async def get_debate(debate_id: str) -> dict:
    session = _session(debate_id)
    return {
        "debate_id": debate_id, "topic": session.topic, "status": session.status,
        "error": session.error,
        "rounds": session.config.rounds,
        "participants": [{"id": s.id, "label": s.label, "model": s.model}
                         for s in session.specs],
        "running": session.running_totals(),
        "events": session.events,
        "waited_s": round(session.gate.total_waited_s, 1),
    }
