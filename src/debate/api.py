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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import DEFAULT_PERSONA_FALLBACK, split_provider_model
from .config import (
    FAKE_PROVIDER, PricingTable, Settings, load_provider_slots,
    provider_registry,
)
from .config import _NAME_RE as _NAME_OK
from .cost import CostMeter, Estimator
from .judge import family_note, vendor_family
from .envfile import SlotInput, mask_key, save_slots
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


# ── 프로바이더 설정 ──────────────────────────────────────────────────────────

#: 루프백으로 인정하는 주소.
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}

#: 프록시를 거쳤다는 표시. 설정 편집은 직접 로컬 접속만 받습니다.
_PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded", "x-forwarded-host")

PROVIDER_PRESETS = [
    {"name": "gemini",
     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
     "키 발급": "https://aistudio.google.com — Get API key"},
    {"name": "groq", "base_url": "https://api.groq.com/openai/v1",
     "키 발급": "https://console.groq.com/keys (무료 티어, 레이트리밋 빡빡)"},
    {"name": "openrouter", "base_url": "https://openrouter.ai/api/v1",
     "키 발급": "https://openrouter.ai/keys (무료 모델은 :free 로 끝남)"},
]


def _require_local(request: Request) -> None:
    """설정 편집은 직접 로컬 접속에서만.

    **바인드 주소로는 판별할 수 없습니다.** `scope["server"]` 는 그 연결의 로컬
    소켓 주소라서, 0.0.0.0 으로 띄우고 localhost 로 접속하면 127.0.0.1 로
    보입니다(실측). 그래서 "외부에서 닿을 수 있는가"가 아니라 "이 요청이 외부에서
    왔는가"를 막습니다 — 실제로 지켜야 할 성질은 그쪽입니다.

    클라이언트 주소는 밖에서 위조할 수 없습니다. 외부 IP 에서
    `X-Forwarded-For: 127.0.0.1` 을 붙여도 uvicorn 이 무시합니다(실측).
    """
    settings = Settings()
    if settings.config_ui.lower() != "local":
        raise HTTPException(404, "설정 UI 가 꺼져 있습니다 (DEBATE_CONFIG_UI)")

    present = [h for h in _PROXY_HEADERS if h in request.headers]
    if present:
        raise HTTPException(403, (
            f"프록시를 거친 요청은 설정을 편집할 수 없습니다 ({', '.join(present)}). "
            "서버가 있는 컴퓨터에서 직접 열어 주십시오."))

    client = request.client.host if request.client else None
    if client not in _LOOPBACK:
        raise HTTPException(403, (
            f"설정 편집은 로컬에서만 가능합니다 (요청 출처: {client}). "
            "이 화면은 .env 를 쓰고 API 키를 다룹니다."))


class SlotIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    base_url: str = Field(min_length=1)
    #: 비우면 **기존 키를 유지**합니다. 화면에는 마스킹만 내려가므로, 수정하지
    #: 않은 슬롯의 키를 브라우저가 되돌려 보낼 방법이 아예 없습니다.
    api_key: str = ""


class SaveSlots(BaseModel):
    slots: list[SlotIn] = Field(max_length=9)


class TestSlot(BaseModel):
    name: str = ""
    base_url: str = Field(min_length=1)
    api_key: str = ""


def _stored_keys() -> dict[str, str]:
    registry = provider_registry()
    out: dict[str, str] = {}
    for name in registry.names():
        slot = registry.get(name)
        out[name] = slot.api_key.get_secret_value() if slot.api_key else ""
    return out


@app.get("/config/providers")
async def config_list(request: Request) -> dict:
    """슬롯 목록. **키는 마스킹만 내려갑니다.**"""
    _require_local(request)
    settings = Settings()
    registry = provider_registry()
    return {
        "env_path": str(settings.env_path.resolve()),
        "env_exists": settings.env_path.is_file(),
        "slots": [
            {"name": name,
             "base_url": registry.get(name).base_url,
             "key_masked": mask_key(
                 registry.get(name).api_key.get_secret_value()
                 if registry.get(name).api_key else None),
             "has_key": registry.get(name).has_key}
            for name in registry.names()
        ],
        "presets": PROVIDER_PRESETS,
    }


@app.post("/config/providers/test")
async def config_test(request: Request, body: TestSlot) -> dict:
    """저장 전 연결 확인. 모델 목록을 가져와 보여줍니다."""
    _require_local(request)
    from pydantic import SecretStr

    from .config import ProviderSlot
    from .provider import OpenAICompatProvider

    key = body.api_key or _stored_keys().get(body.name.strip().lower(), "")
    slot = ProviderSlot(name=body.name or "test",
                        base_url=body.base_url.strip().rstrip("/"),
                        api_key=SecretStr(key) if key else None)
    provider = OpenAICompatProvider(slot, timeout_s=20.0)
    try:
        models = await provider.list_models()
        return {"ok": True, "models": models[:50], "count": len(models),
                "used_stored_key": not body.api_key and bool(key)}
    except Exception as e:                                  # noqa: BLE001
        # 예외 문자열에 키가 섞이지 않게: 프로바이더 에러는 본문 앞부분만 담습니다.
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    finally:
        await provider.aclose()


@app.put("/config/providers")
async def config_save(request: Request, body: SaveSlots) -> dict:
    """.env 를 갱신합니다. BOM 없는 UTF-8, 원자적 교체, 권한 0600."""
    _require_local(request)
    settings = Settings()
    stored = _stored_keys()

    seen: set[str] = set()
    slots: list[SlotInput] = []
    for entry in body.slots:
        name = entry.name.strip().lower()
        if not _NAME_OK.match(name):
            raise HTTPException(400, f"프로바이더 이름 {name!r}: 소문자/숫자/_/- 만 됩니다")
        if name == FAKE_PROVIDER:
            raise HTTPException(400, f"{FAKE_PROVIDER!r} 는 예약어입니다")
        if name in seen:
            raise HTTPException(400, f"프로바이더 이름 중복: {name!r}")
        base_url = entry.base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise HTTPException(400, f"{name}: BASE_URL 은 http(s):// 로 시작해야 합니다")
        seen.add(name)
        # 빈 키 = 기존 유지. 이름이 바뀌었으면 기존 키가 없으므로 빈 값이 됩니다.
        slots.append(SlotInput(name, base_url, entry.api_key or stored.get(name, "")))

    save_slots(settings.env_path, slots)
    registry = provider_registry()      # 바로 다시 읽어 반영 확인
    return {"saved": True, "path": str(settings.env_path.resolve()),
            "names": list(registry.names())}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/models")
async def models(provider: str = FAKE_PROVIDER) -> dict:
    """모델 ID 목록. base_url·키는 응답에 넣지 않습니다."""
    if provider == FAKE_PROVIDER:
        return {"provider": provider, "models": await FakeProvider().list_models()}
    registry = provider_registry()
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
    return {"providers": list(provider_registry().names()), "fake": FAKE_PROVIDER}


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
        provider, model = split_provider_model(body.judge, specs[0].provider)
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
            pool_specs, None if body.use_fake else provider_registry(settings),
            meter, settings,
            fake=_make_fake(body) if body.use_fake else None,
        )
    except ConfigError as e:
        raise HTTPException(400, str(e)) from None

    estimate = Estimator(pricing, settings.ko_tokens_per_char).estimate(
        participants=specs, rounds=body.rounds,
        judge_model=judge_spec[1] if judge_spec else None, topic=body.topic)

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
            "judge_repair_tokens": estimate.judge_repair_tokens,
        },
        "participants": [{"id": s.id, "label": s.label, "model": s.model,
                          "stance": s.stance, "persona": s.persona} for s in specs],
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
async def stream(debate_id: str, request: Request) -> StreamingResponse:
    """SSE. 밀린 이벤트를 재생하고 이후 실시간으로 흘립니다.

    각 이벤트에 `id: <seq>` 를 붙입니다. 브라우저는 재연결 시 마지막 id 를
    Last-Event-ID 헤더로 돌려주므로, 자동 재연결에서는 그 이후만 보냅니다.
    이게 없으면 재연결마다 전체 버퍼가 다시 흘러가 화면이 중복 누적됩니다.
    """
    session = _session(debate_id)
    raw_id = request.headers.get("last-event-id")
    after = int(raw_id) if raw_id and raw_id.lstrip("-").isdigit() else None

    async def gen():
        queue = session.subscribe(after_seq=after)
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"      # 프록시가 끊지 않게
                    continue
                yield (f"id: {payload.get('seq', 0)}\n"
                       f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
                if payload.get("type") == "finished":
                    # 스트림을 닫으면 EventSource 는 그걸 '끊김'으로 보고 다시
                    # 붙습니다(정상 종료여도). 클라이언트가 close() 하도록
                    # 알린 뒤, 재접속해도 커서 이후에는 보낼 게 없습니다.
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
