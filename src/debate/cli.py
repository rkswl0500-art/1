"""터미널 진입점. 슬라이스 1 의 검증 표면입니다.

    python -m debate.cli models --provider fake
    python -m debate.cli run --provider-override fake \
        --topic "..." --agent fake-a --agent fake-b --rounds 1
"""

from __future__ import annotations

import argparse
import asyncio
import string
import sys
from decimal import Decimal
from pathlib import Path

import yaml

from .agent import Agent
from .config import FAKE_PROVIDER, PricingTable, Settings, load_provider_slots
from .cost import CostMeter
from .engine import DebateEngine
from .models import AgentSpec, ConfigError, DebateConfig, DebateResult
from .provider import FakeBehavior, FakeProvider, ProviderError, build_pool

DEFAULT_PERSONA = "논리적 근거를 중시하는 토론자"


def _label(i: int) -> str:
    """0 -> '참가자 A'. 26명 넘으면 AA, AB ... (상한은 5명이라 사실상 A~E)."""
    letters = string.ascii_uppercase
    name = ""
    n = i
    while True:
        name = letters[n % 26] + name
        n = n // 26 - 1
        if n < 0:
            break
    return f"참가자 {name}"


def _parse_agent_flag(raw: str, index: int, *, fake: bool) -> AgentSpec:
    """`provider/model` 또는 (fake 모드에서) `model` 을 AgentSpec 으로."""
    if "/" in raw:
        provider, _, model = raw.partition("/")
    elif fake:
        provider, model = FAKE_PROVIDER, raw
    else:
        raise ConfigError(
            f"--agent {raw!r}: 'provider/model' 형식으로 쓰세요 (예: openai/gpt-4o-mini). "
            "프로바이더 이름은 .env 의 DEBATE_PROVIDER_*_NAME 입니다."
        )
    provider, model = provider.strip().lower(), model.strip()
    if not provider or not model:
        raise ConfigError(f"--agent {raw!r}: provider 와 model 이 모두 필요합니다")
    return AgentSpec(
        id=f"p{index + 1}",
        label=_label(index),
        provider=provider,
        model=model,
        persona=DEFAULT_PERSONA,
    )


def _load_roster(path: Path, *, fake: bool) -> tuple[str, int, list[AgentSpec]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("participants") or []
    if not entries:
        raise ConfigError(f"{path}: participants 가 비어 있습니다")

    specs: list[AgentSpec] = []
    for i, e in enumerate(entries):
        model = str(e.get("model", "")).strip()
        provider = str(e.get("provider", "")).strip().lower()
        if not model or model == "REPLACE_ME":
            raise ConfigError(
                f"{path}: 참가자 {e.get('id', i + 1)} 의 model 이 안 채워졌습니다"
            )
        if "label" in e:
            raise ConfigError(
                f"{path}: label 은 지정할 수 없습니다 (익명 라벨은 자동 배정)"
            )
        specs.append(
            AgentSpec(
                id=str(e.get("id") or f"p{i + 1}"),
                label=_label(i),
                provider=FAKE_PROVIDER if fake else provider,
                model=model,
                persona=str(e.get("persona") or DEFAULT_PERSONA),
                stance=(str(e["stance"]) if e.get("stance") else None),
            )
        )
    return str(raw.get("topic") or ""), int(raw.get("rounds") or 1), specs


def _build_fake(specs: list[AgentSpec], latencies: str | None) -> FakeProvider:
    """--fake-latency 1800,2100 을 참가자 순서대로 모델에 매핑합니다."""
    if not latencies:
        return FakeProvider()
    values = [int(v) for v in latencies.split(",") if v.strip()]
    if len(values) != len(specs):
        raise ConfigError(
            f"--fake-latency 값이 {len(values)}개인데 참가자는 {len(specs)}명입니다"
        )
    return FakeProvider({s.model: FakeBehavior(latency_ms=v) for s, v in zip(specs, values)})


# ── 출력 ─────────────────────────────────────────────────────────────────────


def _n(x: int) -> str:
    return f"{x:,}"


def _usd(x: Decimal) -> str:
    return f"${x.quantize(Decimal('0.0001'))}"


def _print_result(result: DebateResult, meter: CostMeter) -> None:
    labels = {s.id: s.label for s in result.participants}
    models = {s.id: f"{s.provider}/{s.model}" for s in result.participants}

    for rnd in result.rounds:
        print()
        for u in rnd.utterances:
            head = f"[R{rnd.round_no}] {labels[u.agent_id]}"
            if u.status == "failed":
                print(f"{head}  ── 실패: {u.error}")
                continue
            print(f"{head}  ({_n(u.latency_ms)}ms, {models[u.agent_id]})")
            for line in u.content.splitlines() or [""]:
                print(f"  {line}")
            print()

        longest = max((u.latency_ms for u in rnd.utterances), default=0)
        print(
            f"wall {_n(rnd.wall_ms)}ms | sum {_n(rnd.sum_latency_ms)}ms | "
            f"max {_n(longest)}ms | waves {rnd.waves}"
        )

    if result.dropped:
        print(f"dropout: {', '.join(result.dropped)}")
    if result.status != "completed":
        print(f"status: {result.status}")

    rep = meter.report()
    line = (
        f"[cost] calls={rep.calls}  in={_n(rep.prompt_tokens)}  "
        f"out={_n(rep.completion_tokens)}  {_usd(rep.total_usd)}"
    )
    if rep.unpriced_models:
        line += f"  (가격 미상: {', '.join(rep.unpriced_models)} — 0 으로 계상)"
    print(line)


# ── 커맨드 ───────────────────────────────────────────────────────────────────


async def _cmd_models(args: argparse.Namespace) -> int:
    settings = Settings()
    name = args.provider.lower()
    if name == FAKE_PROVIDER:
        for m in await FakeProvider().list_models():
            print(m)
        return 0

    slots = load_provider_slots()
    if name not in slots:
        raise ConfigError(
            f"프로바이더 {name!r} 가 .env 에 없습니다. "
            f"사용 가능: {', '.join(sorted(slots)) or '<없음>'}"
        )
    from .provider import OpenAICompatProvider

    provider = OpenAICompatProvider(slots[name], timeout_s=settings.request_timeout_s)
    try:
        for m in await provider.list_models():
            print(m)
    finally:
        await provider.aclose()
    return 0


async def _cmd_run(args: argparse.Namespace) -> int:
    settings = Settings()
    fake = args.provider_override == FAKE_PROVIDER

    if args.config:
        topic, rounds, specs = _load_roster(Path(args.config), fake=fake)
        topic = args.topic or topic
        rounds = args.rounds if args.rounds is not None else rounds
    else:
        if not args.agent:
            raise ConfigError("--agent 를 최소 2개 주거나 --config 를 쓰세요")
        specs = [_parse_agent_flag(a, i, fake=fake) for i, a in enumerate(args.agent)]
        topic = args.topic or ""
        rounds = args.rounds if args.rounds is not None else 1

    if not topic:
        raise ConfigError("--topic 이 필요합니다")
    if not 2 <= len(specs) <= 5:
        raise ConfigError(f"참가자는 2~5명이어야 합니다 (현재 {len(specs)}명)")

    pricing = PricingTable.load(settings.pricing_path)
    meter = CostMeter(pricing, debate_id="pending")
    pool = build_pool(
        specs,
        {} if fake else load_provider_slots(),
        meter,
        settings,
        fake=_build_fake(specs, args.fake_latency) if fake else None,
    )

    concurrency = args.max_concurrency or settings.max_concurrency
    agents = [
        Agent(s, pool.get(s.provider), pricing, timeout_s=settings.request_timeout_s)
        for s in specs
    ]
    cfg = DebateConfig(
        topic=topic,
        participants=tuple(specs),
        rounds=rounds,
        max_concurrency=concurrency,
        round_timeout_s=args.round_timeout or settings.request_timeout_s + 60,
    )

    print(f"주제: {topic}")
    print(
        f"참가자 {len(specs)}명 | 라운드 {rounds} | 동시성 {concurrency} | "
        f"프로바이더 {'fake' if fake else ', '.join(sorted({s.provider for s in specs}))}"
    )

    try:
        result = await DebateEngine(agents).run(cfg)
    finally:
        await pool.aclose()

    _print_result(result, meter)
    return 0 if result.status == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="debate", description="AI Debate Room")
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("models", help="프로바이더가 노출하는 모델 ID 목록")
    m.add_argument("--provider", default=FAKE_PROVIDER)
    m.set_defaults(fn=_cmd_models)

    r = sub.add_parser("run", help="토론 실행")
    r.add_argument("--topic")
    r.add_argument(
        "--agent", action="append",
        help="'provider/model' (fake 모드면 'model' 만도 가능). 반복 지정.",
    )
    r.add_argument("--config", help="참가자 로스터 yaml")
    r.add_argument("--rounds", type=int)
    r.add_argument("--max-concurrency", type=int)
    r.add_argument("--round-timeout", type=float)
    r.add_argument(
        "--provider-override", choices=[FAKE_PROVIDER],
        help="fake 를 주면 실제 호출 없이 결정론적 가짜로 돌립니다",
    )
    r.add_argument("--fake-latency", help="fake 모드 참가자별 지연(ms), 콤마 구분")
    r.set_defaults(fn=_cmd_run)

    args = parser.parse_args(argv)
    try:
        return asyncio.run(args.fn(args))
    except ConfigError as e:
        print(f"ConfigError: {e}", file=sys.stderr)
        return 2
    except ProviderError as e:
        print(f"ProviderError: {e}", file=sys.stderr)
        return 3
    except NotImplementedError as e:
        print(f"NotImplemented: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
