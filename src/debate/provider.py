"""Provider 레이어. 인터페이스는 ChatProvider 하나뿐입니다.

구현체:
  OpenAICompatProvider  OpenAI 호환 엔드포인트면 무엇이든. base_url 로 결정.
  FakeProvider          결정론적 가짜. 지연/실패를 심을 수 있음.

데코레이터(역시 ChatProvider):
  RetryingProvider      재시도 + 백오프. 429 는 Retry-After 존중.
  MeteredProvider       토큰/비용/지연을 원장에 기록.

재시도와 계측을 구현체가 아니라 데코레이터로 둔 덕분에, 프로바이더를 추가해도
그 두 정책을 다시 짤 필요가 없습니다. 조립은 build_pool() 이 합니다.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import httpx

from .config import FAKE_PROVIDER, ProviderSlot, Settings
from .cost import CostMeter, estimate_tokens
from .models import AgentSpec, ChatRequest, ChatResponse, ConfigError, Usage

# ── 에러 ─────────────────────────────────────────────────────────────────────


class ProviderError(Exception):
    """프로바이더 호출 실패의 최상위."""


class RetryableError(ProviderError):
    """일시적 실패. 재시도 가치가 있음 (5xx, 타임아웃, 커넥션 끊김)."""


class RateLimitError(RetryableError):
    """429. 재시도하되 더 오래 기다립니다."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class FatalError(ProviderError):
    """재시도해도 소용없음 (401, 400, 모델 없음). 즉시 포기."""


# ── 인터페이스 ───────────────────────────────────────────────────────────────


@runtime_checkable
class ChatProvider(Protocol):
    name: str

    async def chat(self, req: ChatRequest) -> ChatResponse: ...

    async def list_models(self) -> list[str]: ...

    async def aclose(self) -> None: ...


# ── 실제 구현체 ──────────────────────────────────────────────────────────────


class OpenAICompatProvider:
    """OpenAI 호환(`/chat/completions`) 엔드포인트 클라이언트.

    base_url 하나만 다르면 OpenAI, Groq, OpenRouter, 로컬 vLLM/Ollama 어디든
    같은 코드로 붙습니다. api_key 가 없으면 Authorization 헤더를 아예 안 붙여
    인증 없는 로컬 엔드포인트도 그대로 지원합니다.
    """

    def __init__(self, slot: ProviderSlot, *, timeout_s: float = 120.0,
                 max_connections: int = 10,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.name = slot.name
        self._base_url = slot.base_url
        headers = {"content-type": "application/json"}
        if slot.has_key:
            assert slot.api_key is not None
            headers["authorization"] = f"Bearer {slot.api_key.get_secret_value()}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout_s,
            limits=httpx.Limits(max_connections=max_connections),
            transport=transport,  # 테스트에서 MockTransport 주입용
        )

    async def chat(self, req: ChatRequest) -> ChatResponse:
        payload: dict = {
            "model": req.model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "temperature": req.temperature,
        }
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens

        started = time.perf_counter()
        try:
            resp = await self._client.post(
                "/chat/completions", json=payload, timeout=req.timeout_s
            )
        except httpx.TimeoutException as e:
            raise RetryableError(f"{self.name}: timeout ({e})") from e
        except httpx.TransportError as e:
            raise RetryableError(f"{self.name}: transport error ({e})") from e
        latency_ms = int((time.perf_counter() - started) * 1000)

        self._raise_for_status(resp)

        try:
            body = resp.json()
            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise RetryableError(f"{self.name}: 응답 형식이 예상과 다름 ({e})") from e

        raw_usage = body.get("usage") or {}
        return ChatResponse(
            text=text,
            model=body.get("model") or req.model,
            usage=Usage(
                prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
                completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            ),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
        )

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        snippet = resp.text[:300]
        if resp.status_code == 429:
            raw = resp.headers.get("retry-after")
            retry_after: float | None = None
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
            raise RateLimitError(f"{self.name}: 429 rate limited — {snippet}", retry_after)
        if resp.status_code >= 500:
            raise RetryableError(f"{self.name}: {resp.status_code} — {snippet}")
        raise FatalError(f"{self.name}: {resp.status_code} — {snippet}")

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/models")
        except httpx.HTTPError as e:
            raise RetryableError(f"{self.name}: /models 실패 ({e})") from e
        self._raise_for_status(resp)
        body = resp.json()
        return [str(m.get("id")) for m in body.get("data", []) if m.get("id")]

    async def aclose(self) -> None:
        await self._client.aclose()


# ── Fake ─────────────────────────────────────────────────────────────────────

FailMode = Literal["500", "429", "timeout", "fatal"]


@dataclass(frozen=True, slots=True)
class FakeBehavior:
    latency_ms: int = 800
    #: 앞의 N 번 호출을 실패시킵니다. 재시도 경로 검증용.
    fail_first_n: int = 0
    fail_mode: FailMode = "500"
    fail_always: bool = False


_FAKE_OPENERS = (
    "이 주제에서 결정적인 변수는 측정 방식입니다.",
    "반대편이 전제로 삼는 인과관계부터 의심해야 합니다.",
    "통계보다 먼저 정의를 합의해야 논쟁이 성립합니다.",
    "단기 효과와 장기 효과를 섞어 말하면 결론이 왜곡됩니다.",
    "비용을 누가 부담하는지를 빼놓으면 절반만 본 것입니다.",
)
_FAKE_BODIES = (
    "관측된 지표가 실제 성과를 대리한다는 보장이 없고, 대리변수가 어긋나면 결론 전체가 흔들립니다.",
    "제도를 도입한 조직과 도입하지 않은 조직은 애초에 특성이 다르므로 선택편향을 통제해야 합니다.",
    "전환 비용은 초기에 집중되고 편익은 뒤늦게 나타나므로 관측 구간에 따라 부호가 뒤집힙니다.",
    "평균값 뒤에 분산이 숨어 있어, 특정 직군에서만 성립하는 효과가 전체 효과로 포장됩니다.",
    "반대 사례가 존재한다는 사실만으로는 일반화를 무너뜨리지 못하며 빈도를 따져야 합니다.",
)
_FAKE_CLOSERS = (
    "따라서 저는 조건부로만 이 주장에 동의합니다.",
    "그러므로 입증 책임은 여전히 반대편에 있습니다.",
    "이 지점이 해소되기 전까지 결론을 유보해야 합니다.",
)


class FakeProvider:
    """결정론적 가짜 프로바이더.

    테스트 픽스처가 아니라 1급 실행 모드입니다. 병렬성/재시도/dropout/컨텍스트
    구성 같은 항목은 실제 모델보다 여기서 **더 정확하게** 검증됩니다 — 재현이
    되니까요. 같은 입력이면 항상 같은 출력이 나옵니다.
    """

    def __init__(
        self,
        behaviors: Mapping[str, FakeBehavior] | None = None,
        default: FakeBehavior | None = None,
    ) -> None:
        self.name = FAKE_PROVIDER
        self._behaviors = dict(behaviors or {})
        self._default = default or FakeBehavior()
        self._calls: dict[str, int] = {}

    def behavior_for(self, model: str) -> FakeBehavior:
        return self._behaviors.get(model, self._default)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        b = self.behavior_for(req.model)
        seen = self._calls.get(req.model, 0)
        self._calls[req.model] = seen + 1

        if b.fail_always or seen < b.fail_first_n:
            await asyncio.sleep(min(b.latency_ms, 50) / 1000)
            raise _fake_failure(self.name, req.model, b.fail_mode)

        await asyncio.sleep(b.latency_ms / 1000)

        prompt_text = "\n".join(m.content for m in req.messages)
        text = self._compose(req, prompt_text)
        return ChatResponse(
            text=text,
            model=req.model,
            usage=Usage(
                prompt_tokens=estimate_tokens(prompt_text),
                completion_tokens=estimate_tokens(text),
            ),
            latency_ms=b.latency_ms,
            finish_reason="stop",
        )

    def _compose(self, req: ChatRequest, prompt_text: str) -> str:
        digest = hashlib.sha256(f"{req.model}|{prompt_text}".encode()).digest()
        pick = lambda seq, i: seq[digest[i] % len(seq)]  # noqa: E731
        return (
            f"{pick(_FAKE_OPENERS, 0)} "
            f"{pick(_FAKE_BODIES, 1)} "
            f"{pick(_FAKE_BODIES, 2)} "
            f"{pick(_FAKE_CLOSERS, 3)}"
        )

    async def list_models(self) -> list[str]:
        return sorted({*self._behaviors, "fake-a", "fake-b"})

    async def aclose(self) -> None:
        return None


def _fake_failure(provider: str, model: str, mode: FailMode) -> ProviderError:
    if mode == "429":
        return RateLimitError(f"{provider}/{model}: 주입된 429", retry_after_s=0.01)
    if mode == "timeout":
        return RetryableError(f"{provider}/{model}: 주입된 timeout")
    if mode == "fatal":
        return FatalError(f"{provider}/{model}: 주입된 401")
    return RetryableError(f"{provider}/{model}: 주입된 500")


# ── 데코레이터 ───────────────────────────────────────────────────────────────


class RetryingProvider:
    """재시도 + 지수 백오프. 감싼 대상도 ChatProvider 이므로 무엇이든 감쌉니다.

    429 는 일반 5xx 와 다르게 취급합니다. 무료 프로바이더는 레이트리밋이 빡빡해
    1s/2s/4s 로는 부족하고, Retry-After 를 주면 그 값을 그대로 따릅니다.
    FatalError 는 재시도하지 않고 즉시 올려보냅니다.
    """

    def __init__(
        self,
        inner: ChatProvider,
        *,
        attempts: int = 3,
        base_delay_s: float = 1.0,
        rate_limit_base_delay_s: float = 4.0,
        max_delay_s: float = 30.0,
        sleep=asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._inner = inner
        self.name = inner.name
        self._attempts = max(1, attempts)
        self._base = base_delay_s
        self._rl_base = rate_limit_base_delay_s
        self._max = max_delay_s
        self._sleep = sleep
        self._rng = rng or random.Random(0)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        last: ProviderError | None = None

        for attempt in range(1, self._attempts + 1):
            try:
                resp = await self._inner.chat(req)
            except FatalError:
                raise  # 재시도 무의미
            except RetryableError as e:
                last = e
                if attempt == self._attempts:
                    break
                await self._sleep(self._delay_for(e, attempt))
                continue
            return replace(
                resp,
                latency_ms=int((time.perf_counter() - started) * 1000),
                attempts=attempt,
            )

        assert last is not None
        raise RetryableError(
            f"{self.name}: {self._attempts}회 시도 모두 실패 — {last}"
        ) from last

    def _delay_for(self, err: RetryableError, attempt: int) -> float:
        if isinstance(err, RateLimitError) and err.retry_after_s is not None:
            return min(err.retry_after_s, self._max)
        base = self._rl_base if isinstance(err, RateLimitError) else self._base
        backoff = min(base * (2 ** (attempt - 1)), self._max)
        return backoff * (0.8 + 0.4 * self._rng.random())  # 지터

    async def list_models(self) -> list[str]:
        return await self._inner.list_models()

    async def aclose(self) -> None:
        await self._inner.aclose()


class MeteredProvider:
    """모든 호출의 토큰/비용/지연을 원장에 남깁니다."""

    def __init__(self, inner: ChatProvider, meter: CostMeter, purpose: str = "debate") -> None:
        self._inner = inner
        self._meter = meter
        self._purpose = purpose
        self.name = inner.name

    async def chat(self, req: ChatRequest) -> ChatResponse:
        resp = await self._inner.chat(req)
        self._meter.record(
            purpose=self._purpose,
            agent_id=req.tag,
            provider=self.name,
            model=resp.model,
            usage=resp.usage,
            latency_ms=resp.latency_ms,
            attempts=resp.attempts,
            finish_reason=resp.finish_reason,
        )
        return resp

    async def list_models(self) -> list[str]:
        return await self._inner.list_models()

    async def aclose(self) -> None:
        await self._inner.aclose()


# ── 조립 ─────────────────────────────────────────────────────────────────────


class ProviderPool:
    """프로바이더 이름 -> 조립된 ChatProvider."""

    def __init__(self, providers: Mapping[str, ChatProvider]) -> None:
        self._providers = dict(providers)

    def get(self, name: str) -> ChatProvider:
        try:
            return self._providers[name]
        except KeyError:
            available = ", ".join(sorted(self._providers)) or "<없음>"
            raise ConfigError(
                f"프로바이더 {name!r} 가 설정되어 있지 않습니다. "
                f"사용 가능: {available}. .env 의 DEBATE_PROVIDER_*_NAME 을 확인하세요."
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    async def aclose(self) -> None:
        for p in self._providers.values():
            await p.aclose()


def build_pool(
    specs: Sequence[AgentSpec],
    slots: Mapping[str, ProviderSlot],
    meter: CostMeter,
    settings: Settings,
    *,
    fake: FakeProvider | None = None,
) -> ProviderPool:
    """참가자들이 실제로 참조하는 프로바이더만 조립합니다.

    fake 가 주어지면 참가자의 provider 필드와 무관하게 전부 fake 로 흘립니다
    (슬라이스 1 검증 모드). 그 외에는 이름별로 슬롯을 찾아 조립하고, 없는
    이름을 참조하면 LLM 을 한 번도 부르기 전에 ConfigError 로 죽습니다.
    """
    wanted = {s.provider for s in specs}

    if fake is not None:
        wrapped = MeteredProvider(
            RetryingProvider(fake, attempts=settings.retry_attempts), meter
        )
        return ProviderPool({name: wrapped for name in wanted | {FAKE_PROVIDER}})

    missing = sorted(wanted - set(slots))
    if missing:
        available = ", ".join(sorted(slots)) or "<없음>"
        raise ConfigError(
            f"참가자가 참조하는 프로바이더가 .env 에 없습니다: {', '.join(missing)}. "
            f"사용 가능: {available}"
        )

    built: dict[str, ChatProvider] = {}
    for name in sorted(wanted):
        raw = OpenAICompatProvider(
            slots[name],
            timeout_s=settings.request_timeout_s,
            max_connections=max(settings.max_concurrency, 1),
        )
        built[name] = MeteredProvider(
            RetryingProvider(raw, attempts=settings.retry_attempts), meter
        )
    return ProviderPool(built)
