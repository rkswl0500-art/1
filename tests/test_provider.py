"""Provider 레이어 테스트. 실제 네트워크를 쓰지 않습니다."""

from __future__ import annotations

from decimal import Decimal

import httpx
import pytest

from debate.config import PricingTable, ProviderSlot, load_provider_slots
from debate.cost import CostMeter
from debate.models import ChatRequest, ChatResponse, ConfigError, Message, Usage
from debate.provider import (
    FakeBehavior,
    FakeProvider,
    FatalError,
    MeteredProvider,
    OpenAICompatProvider,
    RateLimitError,
    RetryableError,
    RetryingProvider,
)
from pydantic import SecretStr

REQ = ChatRequest(model="m", messages=(Message("user", "안녕"),), tag="p1")


async def _no_sleep(_: float) -> None:
    return None


def _meter() -> CostMeter:
    return CostMeter(PricingTable({}), "d_test")


# ── 재시도 ───────────────────────────────────────────────────────────────────


async def test_retries_then_succeeds_and_reports_attempts():
    inner = FakeProvider({"m": FakeBehavior(latency_ms=1, fail_first_n=2)})
    p = RetryingProvider(inner, attempts=3, sleep=_no_sleep)
    resp = await p.chat(REQ)
    assert resp.attempts == 3


async def test_gives_up_after_configured_attempts():
    inner = FakeProvider({"m": FakeBehavior(latency_ms=1, fail_always=True)})
    p = RetryingProvider(inner, attempts=3, sleep=_no_sleep)
    with pytest.raises(RetryableError, match="3회 시도 모두 실패"):
        await p.chat(REQ)


async def test_fatal_error_is_not_retried():
    """401 을 3번 다시 보내는 건 낭비이자 계정 잠금 위험입니다."""
    calls = 0

    class Boom:
        name = "boom"

        async def chat(self, req):
            nonlocal calls
            calls += 1
            raise FatalError("401")

        async def list_models(self): return []
        async def aclose(self): return None

    with pytest.raises(FatalError):
        await RetryingProvider(Boom(), attempts=3, sleep=_no_sleep).chat(REQ)
    assert calls == 1


async def test_rate_limit_honors_retry_after():
    slept: list[float] = []

    async def spy(d: float) -> None:
        slept.append(d)

    inner = FakeProvider({"m": FakeBehavior(latency_ms=1, fail_first_n=1, fail_mode="429")})
    await RetryingProvider(inner, attempts=3, sleep=spy).chat(REQ)
    assert slept == [0.01]  # 서버가 준 Retry-After 를 그대로 따름


async def test_rate_limit_backoff_is_longer_than_5xx():
    """무료 프로바이더 대응: 429 는 5xx 보다 길게 기다립니다."""
    p = RetryingProvider(FakeProvider(), base_delay_s=1.0, rate_limit_base_delay_s=4.0)
    assert p._delay_for(RetryableError("500"), 1) < p._delay_for(RateLimitError("429"), 1)


# ── 계측 ─────────────────────────────────────────────────────────────────────


async def test_metered_records_every_call():
    meter = _meter()
    p = MeteredProvider(FakeProvider({"m": FakeBehavior(latency_ms=1)}), meter)
    await p.chat(REQ)
    await p.chat(REQ)
    rep = meter.report()
    assert rep.calls == 2
    assert rep.prompt_tokens > 0 and rep.completion_tokens > 0
    assert meter.records[0].agent_id == "p1"  # tag 로 참가자에 귀속


async def test_unpriced_model_costs_zero_not_crash():
    meter = CostMeter(PricingTable({}), "d")
    await MeteredProvider(FakeProvider({"m": FakeBehavior(latency_ms=1)}), meter).chat(REQ)
    rep = meter.report()
    assert rep.total_usd == Decimal(0)
    assert "m" in rep.unpriced_models


# ── HTTP 클라이언트 ──────────────────────────────────────────────────────────


def _provider(handler, *, key: str | None = "sk-secret") -> OpenAICompatProvider:
    slot = ProviderSlot("t", "https://x.test/v1", SecretStr(key) if key else None)
    return OpenAICompatProvider(slot, transport=httpx.MockTransport(handler))


async def test_posts_to_chat_completions_and_parses_usage():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "model": "served-model",
            "choices": [{"message": {"content": "답"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        })

    resp = await _provider(handler).chat(REQ)
    assert seen["url"] == "https://x.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret"
    assert resp.text == "답"
    assert resp.model == "served-model"  # 실제 서빙된 모델을 기록
    assert resp.usage == Usage(11, 7)


async def test_no_auth_header_when_key_absent():
    """인증 없는 로컬 엔드포인트도 그대로 지원돼야 합니다."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "x"}}], "usage": {}})

    await _provider(handler, key=None).chat(REQ)
    assert seen["auth"] is None


@pytest.mark.parametrize(
    "status,expected",
    [(429, RateLimitError), (500, RetryableError), (503, RetryableError),
     (401, FatalError), (400, FatalError), (404, FatalError)],
)
async def test_status_classification(status, expected):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="nope")

    with pytest.raises(expected):
        await _provider(handler).chat(REQ)


async def test_malformed_body_is_retryable_not_crash():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(RetryableError):
        await _provider(handler).chat(REQ)


# ── 설정 ─────────────────────────────────────────────────────────────────────


def test_slots_beyond_the_three_examples_are_picked_up():
    slots = load_provider_slots({
        "DEBATE_PROVIDER_1_NAME": "a", "DEBATE_PROVIDER_1_BASE_URL": "https://a/v1",
        "DEBATE_PROVIDER_7_NAME": "g", "DEBATE_PROVIDER_7_BASE_URL": "http://localhost:1/v1",
    })
    assert set(slots) == {"a", "g"}


def test_api_key_never_appears_in_repr():
    slot = ProviderSlot("a", "https://a/v1", SecretStr("sk-topsecret"))
    assert "sk-topsecret" not in repr(slot)
    assert "sk-topsecret" not in str(slot)


@pytest.mark.parametrize("env,match", [
    ({"DEBATE_PROVIDER_1_NAME": "a"}, "함께 채워야"),
    ({"DEBATE_PROVIDER_1_NAME": "fake", "DEBATE_PROVIDER_1_BASE_URL": "https://a/v1"}, "예약어"),
    ({"DEBATE_PROVIDER_1_NAME": "a", "DEBATE_PROVIDER_1_BASE_URL": "ftp://a"}, "http"),
])
def test_bad_slots_rejected(env, match):
    with pytest.raises(ConfigError, match=match):
        load_provider_slots(env)


def test_known_zero_price_differs_from_unknown_price():
    """무료 모델(0 이라고 앎)과 가격 미상(모름)은 구분돼야 합니다."""
    from debate.config import ModelPrice

    t = PricingTable({"free": ModelPrice(Decimal(0), Decimal(0))})
    assert t.cost_for("free", Usage(100, 100)) == (Decimal(0), True)
    assert t.cost_for("mystery", Usage(100, 100)) == (Decimal(0), False)
    assert t.unpriced_models == ("mystery",)


# ── base_url 정규화 (README 의 흔한 실패 항목) ───────────────────────────────


@pytest.mark.parametrize("raw", ["https://x.test/v1", "https://x.test/v1/"])
async def test_trailing_slash_in_base_url_is_harmless(raw):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}], "usage": {}})

    slot = load_provider_slots(
        {"DEBATE_PROVIDER_1_NAME": "t", "DEBATE_PROVIDER_1_BASE_URL": raw}
    )["t"]
    p = OpenAICompatProvider(slot, transport=httpx.MockTransport(handler))
    await p.chat(REQ)
    assert seen["url"] == "https://x.test/v1/chat/completions"


async def test_base_url_missing_v1_produces_wrong_path():
    """BASE_URL 에 /v1 을 빠뜨리면 404 가 납니다. 문서에 적어둔 실패 원인이
    실제로 그렇게 동작하는지 고정해 둡니다."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}], "usage": {}})

    slot = load_provider_slots(
        {"DEBATE_PROVIDER_1_NAME": "t", "DEBATE_PROVIDER_1_BASE_URL": "https://x.test"}
    )["t"]
    await OpenAICompatProvider(slot, transport=httpx.MockTransport(handler)).chat(REQ)
    assert seen["url"] == "https://x.test/chat/completions"  # /v1 이 없음


async def test_response_without_usage_yields_zero_tokens():
    """usage 를 안 주는 엔드포인트가 있습니다. 죽지는 않지만 집계가 0 이 되므로
    CLI 가 이를 감지해 경고해야 합니다(test_cli 참조)."""
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "답"}}]})

    resp = await _provider(handler).chat(REQ)
    assert resp.usage == Usage(0, 0)
    assert resp.text == "답"
