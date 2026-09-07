"""설정 로딩 한 곳.

  - Settings:        .env / 환경변수. 키는 전부 SecretStr.
  - ProviderSlot:    base_url + api_key 한 쌍. 참가자가 이름으로 참조합니다.
  - PricingTable:    config/pricing.yaml. 가격 미상 모델은 0 + 경고, 예외 없음.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import ConfigError, Usage

#: .env 에서 스캔할 프로바이더 슬롯 번호 범위. .env.example 은 3개만 보여주지만
#: 코드가 3개로 제한하지는 않습니다 — _4, _5 를 추가하면 그냥 잡힙니다.
_SLOT_RANGE = range(1, 10)

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

#: fake 모드 전용 예약어. 슬롯으로 등록할 수 없습니다.
FAKE_PROVIDER = "fake"


@dataclass(frozen=True, slots=True)
class ProviderSlot:
    name: str
    base_url: str
    api_key: SecretStr | None

    @property
    def has_key(self) -> bool:
        return self.api_key is not None and bool(self.api_key.get_secret_value())

    def __repr__(self) -> str:  # 키가 로그/트레이스백에 새지 않도록
        return f"ProviderSlot(name={self.name!r}, base_url={self.base_url!r}, has_key={self.has_key})"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="DEBATE_", extra="ignore", case_sensitive=False
    )

    max_concurrency: int = 3
    request_timeout_s: float = 120.0
    retry_attempts: int = 3
    pricing_path: Path = Path("config/pricing.yaml")
    db_path: Path = Path("data/debates.db")
    ko_tokens_per_char: float = 1.0


def load_provider_slots(env: dict[str, str] | None = None) -> dict[str, ProviderSlot]:
    """DEBATE_PROVIDER_{n}_{NAME,BASE_URL,API_KEY} 를 읽어 레지스트리를 만듭니다.

    NAME 과 BASE_URL 이 모두 있는 슬롯만 등록합니다. API_KEY 는 비어도 됩니다
    (인증 없는 로컬 엔드포인트). 슬롯이 하나도 없어도 예외를 던지지 않습니다 —
    fake 모드는 슬롯 없이 돌아야 하고, 실제 호출 시점에 참가자가 존재하지 않는
    프로바이더를 가리키면 그때 ConfigError 가 납니다.
    """
    src = os.environ if env is None else env
    slots: dict[str, ProviderSlot] = {}

    for n in _SLOT_RANGE:
        name = (src.get(f"DEBATE_PROVIDER_{n}_NAME") or "").strip().lower()
        base_url = (src.get(f"DEBATE_PROVIDER_{n}_BASE_URL") or "").strip()
        raw_key = (src.get(f"DEBATE_PROVIDER_{n}_API_KEY") or "").strip()

        if not name and not base_url:
            continue
        if not name or not base_url:
            raise ConfigError(
                f"프로바이더 슬롯 {n}: NAME 과 BASE_URL 은 함께 채워야 합니다 "
                f"(NAME={name or '<빈값>'}, BASE_URL={base_url or '<빈값>'})"
            )
        if not _NAME_RE.match(name):
            raise ConfigError(
                f"프로바이더 슬롯 {n}: NAME 은 소문자/숫자/_/- 만 됩니다: {name!r}"
            )
        if name == FAKE_PROVIDER:
            raise ConfigError(f"프로바이더 슬롯 {n}: {FAKE_PROVIDER!r} 는 예약어입니다")
        if name in slots:
            raise ConfigError(f"프로바이더 이름 중복: {name!r}")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"프로바이더 {name!r}: BASE_URL 은 http(s):// 로 시작해야 합니다")

        slots[name] = ProviderSlot(
            name=name,
            base_url=base_url.rstrip("/"),
            api_key=SecretStr(raw_key) if raw_key else None,
        )

    return slots


# ── 가격표 ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_1m: Decimal
    output_per_1m: Decimal


_MILLION = Decimal(1_000_000)


class PricingTable:
    """모델 단가. 모르는 모델은 0 + 경고이고 절대 예외를 던지지 않습니다.

    무료 프로바이더를 섞으면 가격표에 0 인 모델이 정상적으로 존재합니다.
    "0 이라고 적혀 있음"(안다)과 "표에 없음"(모른다)은 구분됩니다 — 후자만
    unpriced 로 경고합니다.
    """

    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        self._prices = prices or {}
        self._unpriced: set[str] = set()

    @classmethod
    def load(cls, path: Path | str) -> "PricingTable":
        p = Path(path)
        if not p.exists():
            return cls({})  # 가격표가 없어도 토론은 돌아야 합니다
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        out: dict[str, ModelPrice] = {}
        for model, spec in (raw.get("models") or {}).items():
            spec = spec or {}
            out[str(model)] = ModelPrice(
                input_per_1m=Decimal(str(spec.get("input_per_1m", 0))),
                output_per_1m=Decimal(str(spec.get("output_per_1m", 0))),
            )
        return cls(out)

    def cost_for(self, model: str, usage: Usage) -> tuple[Decimal, bool]:
        """(비용, 가격을 알고 있었는가) 를 돌려줍니다."""
        price = self._prices.get(model)
        if price is None:
            self._unpriced.add(model)
            return Decimal(0), False
        cost = (
            Decimal(usage.prompt_tokens) * price.input_per_1m
            + Decimal(usage.completion_tokens) * price.output_per_1m
        ) / _MILLION
        return cost, True

    @property
    def unpriced_models(self) -> tuple[str, ...]:
        return tuple(sorted(self._unpriced))
