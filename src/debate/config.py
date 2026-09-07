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
from typing import Mapping

import yaml
from dotenv import dotenv_values
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
        env_file=".env",
        # PowerShell 의 `Out-File -Encoding utf8` 은 기본으로 BOM 을 붙입니다.
        # utf-8-sig 는 BOM 이 있든 없든 읽습니다.
        env_file_encoding="utf-8-sig",
        env_prefix="DEBATE_",
        extra="ignore",
        case_sensitive=False,
    )

    max_concurrency: int = 3
    request_timeout_s: float = 120.0
    retry_attempts: int = 3
    pricing_path: Path = Path("config/pricing.yaml")
    db_path: Path = Path("data/debates.db")
    ko_tokens_per_char: float = 1.0


#: 기본 .env 경로. env_file=None 을 명시하면 파일을 아예 안 읽습니다.
DEFAULT_ENV_FILE = Path(".env")
_UNSET: object = object()


@dataclass(frozen=True, slots=True)
class ProviderRegistry:
    """프로바이더 슬롯 + **그것을 어디서 읽었는지**.

    출처를 함께 들고 다니는 이유는 에러 메시지 때문입니다. "사용 가능: <없음>"
    만으로는 .env 를 안 읽은 건지, 파일이 없는 건지, 오타인지 구분할 수 없습니다.
    """

    slots: Mapping[str, ProviderSlot]
    env_file: Path | None
    env_file_found: bool

    def __contains__(self, name: object) -> bool:
        return name in self.slots

    def __len__(self) -> int:
        return len(self.slots)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.slots))

    def get(self, name: str) -> ProviderSlot:
        return self.slots[name]

    def source_hint(self) -> str:
        """에러에 붙일 진단. 어디를 봤고 뭘 찾았는지 있는 그대로 적습니다."""
        lines = [f"사용 가능: {', '.join(self.names()) or '<없음>'}"]
        if self.env_file is None:
            lines.append("(.env 를 읽지 않도록 지정된 호출입니다)")
        elif self.env_file_found:
            lines.append(f"읽은 .env: {self.env_file}  (슬롯 {len(self.slots)}개 인식)")
        else:
            lines.append(f"읽으려 한 .env: {self.env_file}  ← 이 경로에 파일이 없습니다")
            lines.append(f"현재 디렉터리: {Path.cwd()}")
        lines.append(
            "확인: DEBATE_PROVIDER_1_NAME 과 DEBATE_PROVIDER_1_BASE_URL 이 "
            "둘 다 채워져 있어야 슬롯으로 인식됩니다."
        )
        return "\n  ".join(lines)


def load_provider_slots(
    env: Mapping[str, str] | None = None,
    env_file: Path | str | None = _UNSET,  # type: ignore[assignment]
) -> ProviderRegistry:
    """.env 와 프로세스 환경변수를 병합해 프로바이더 레지스트리를 만듭니다.

    **왜 .env 를 직접 읽는가**: pydantic-settings 의 `env_file` 은 Settings 의
    선언된 필드만 채우고 `os.environ` 에는 아무것도 주입하지 않습니다. 슬롯은
    개수가 가변이라 Settings 필드로 선언할 수 없으므로, 여기서 파일을 따로
    읽어야 합니다. 이걸 빠뜨려서 .env 에 정확히 써도 슬롯이 0개로 나왔습니다.

    우선순위는 `os.environ` > `.env` 입니다. CI 나 셸에서 준 값이 파일을 이깁니다.

    인코딩은 utf-8-sig 로 읽습니다. PowerShell 의 `Out-File -Encoding utf8` 이
    기본으로 BOM 을 붙이는데, 그러면 첫 줄 키가 `\ufeffDEBATE_...` 가 되어
    조용히 인식되지 않습니다.
    """
    resolved: Path | None
    if env_file is _UNSET:
        resolved = DEFAULT_ENV_FILE
    elif env_file is None:
        resolved = None
    else:
        resolved = Path(env_file)

    from_file: dict[str, str | None] = {}
    found = False
    if resolved is not None:
        resolved = resolved.expanduser()
        found = resolved.is_file()
        if found:
            from_file = dotenv_values(resolved, encoding="utf-8-sig")
        resolved = resolved.absolute()

    process_env = os.environ if env is None else env
    merged: dict[str, str] = {k: v for k, v in from_file.items() if v is not None}
    merged.update(process_env)  # os.environ 이 .env 를 이깁니다

    slots: dict[str, ProviderSlot] = {}
    for n in _SLOT_RANGE:
        name = (merged.get(f"DEBATE_PROVIDER_{n}_NAME") or "").strip().lower()
        base_url = (merged.get(f"DEBATE_PROVIDER_{n}_BASE_URL") or "").strip()
        raw_key = (merged.get(f"DEBATE_PROVIDER_{n}_API_KEY") or "").strip()

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

    return ProviderRegistry(slots=slots, env_file=resolved, env_file_found=found)


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
