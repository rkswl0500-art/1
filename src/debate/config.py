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
    request_timeout_s: float = 120.0   # 관측된 최대 지연 54s 의 2.2배
    retry_attempts: int = 3
    pricing_path: Path = Path("config/pricing.yaml")
    db_path: Path = Path("data/debates.db")
    #: 한국어 토큰 추정 계수.
    #:
    #: 2026-09 Gemini 라이브 실측으로 보정했습니다: 172~319 토큰 / 250~450자
    #: → 0.69~0.71. 처음 1.0 은 보수적 추측이었고 약 1.4배 과대였습니다.
    #: 모델(토크나이저)마다 달라지므로 다른 벤더를 붙이면 다시 재야 합니다.
    ko_tokens_per_char: float = 0.7
    #: 발언 1건의 출력 토큰 상한.
    #:
    #: 한국어는 600자만 해도 토크나이저에 따라 600~900 토큰입니다. 여기에
    #: thinking 계열 모델은 사고 토큰까지 이 예산에서 쓰는 경우가 있어, 예산이
    #: 빠듯하면 본문이 중간에 끊깁니다. 토론에서 잘린 발언은 곧 잘못된 판정이라
    #: 넉넉하게 잡습니다.
    max_output_tokens: int = 2048
    #: 참가자 1명이 한 라운드에서 쓸 수 있는 총 시간(재시도 포함).
    #:
    #: 이 값이 retry_attempts × request_timeout_s 보다 작으면, 호출이 실제로
    #: 타임아웃까지 매달릴 때 재시도 예산을 다 쓰기 전에 잘립니다. 둘 다 크게
    #: 잡으면 느린 참가자 하나가 라운드를 몇 분씩 붙잡으므로 트레이드오프입니다.
    #: 기본값은 retry_attempts × request_timeout_s (=360s) 를 덮도록 잡았습니다.
    #: 이보다 낮추면 호출이 매달릴 때 재시도가 잘리고, 그 사실이 조용히 넘어가지
    #: 않도록 timeout_warning() 이 알려줍니다.
    #:
    #: 이 값이 커도 정상 동작에는 영향이 없습니다 — 라운드는 실제 지연(관측 54s)
    #: 만큼만 걸리고, 타임아웃은 실패할 때만 물립니다. 다만 진짜로 멈춘 참가자
    #: 하나가 최대 이만큼 라운드를 붙잡을 수 있으므로, 그동안 누구를 기다리는지는
    #: 진행 표시로 보입니다.
    round_timeout_s: float = 400.0
    #: 판정 1건의 전체 시한(재시도 포함).
    #:
    #: 재시도 예산은 기본값에서 6분인데, 심판이 무응답이면 그 동안 화면에
    #: 아무 이벤트도 안 나가 멈춘 것처럼 보입니다. 판정은 단일 호출이라
    #: 토론만큼 오래 걸릴 이유가 없어 더 짧게 끊습니다.
    #: 라이브에서 사고형 심판 모델이 3~4분 걸렸습니다. 180초면 정상 판정을
    #: 중간에 끊습니다. 출력 길이를 잡으면 실제로는 훨씬 빨라질 것으로 보지만,
    #: 그 전에 멀쩡한 판정을 죽이지 않도록 여유를 둡니다. 기다리는 동안
    #: 화면에는 "판정 중"과 경과 시간이 표시됩니다.
    judge_timeout_s: float = 300.0
    #: 0 이면 쟁점·참가자 수에 맞춰 자동 계산합니다(judge.required_output_tokens).
    judge_max_tokens: int = 0
    #: 프로바이더 설정 UI. "local" 이면 루프백 클라이언트만, "off" 면 완전 차단.
    #:
    #: 이 엔드포인트는 .env 를 쓰고 키를 다룹니다. 여러 기기에서 접근하는 구성으로
    #: 옮길 때는 반드시 "off" 로 두십시오.
    config_ui: str = "local"
    env_path: Path = Path(".env")

    @property
    def retry_budget_s(self) -> float:
        return self.retry_attempts * self.request_timeout_s

    def timeout_warning(self) -> str | None:
        """재시도 예산과 라운드 타임아웃이 어긋나면 그 사실을 문장으로."""
        if self.round_timeout_s >= self.retry_budget_s:
            return None
        return (
            f"라운드 타임아웃 {self.round_timeout_s:.0f}s < 재시도 예산 "
            f"{self.retry_budget_s:.0f}s ({self.retry_attempts}회 × "
            f"{self.request_timeout_s:.0f}s). 호출이 타임아웃까지 매달리면 "
            f"재시도를 다 쓰기 전에 라운드 타임아웃이 먼저 참가자를 드롭합니다."
        )


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


def provider_registry(settings: "Settings | None" = None) -> ProviderRegistry:
    """설정이 가리키는 .env 에서 슬롯을 읽습니다.

    `load_provider_slots()` 를 그냥 부르면 항상 CWD 의 `.env` 를 봅니다. 설정 UI 가
    `env_path` 에 쓰는데 읽기는 다른 곳을 보면 방금 저장한 슬롯이 안 보입니다 —
    실제로 그렇게 어긋나 있었습니다. 쓰는 곳과 읽는 곳을 한 값으로 묶습니다.
    """
    s = settings or Settings()
    return load_provider_slots(env_file=s.env_path)


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

    def _lookup(self, model: str) -> "ModelPrice | None":
        """접두사가 붙어 있든 없든 찾습니다.

        같은 모델이 요청과 응답에서 다르게 옵니다 — `models/gemini-3.6-flash` 와
        `gemini-3.6-flash`, `openai/gpt-oss-120b` 와 `gpt-oss-120b`. 접두사 하나
        때문에 "가격 미상" 경고가 뜨면 그 경고는 신호 구실을 못 합니다.

        벤더 접두사를 목록으로 두지 않고 **마지막 구간이 일치하는 키**를 찾습니다.
        후보가 둘 이상이면 포기합니다 — 엉뚱한 단가를 붙이느니 모른다고 하는
        편이 낫습니다.
        """
        price = self._prices.get(model)
        if price is not None:
            return price

        tail = model.rsplit("/", 1)[-1]
        matches = [p for key, p in self._prices.items()
                   if key == tail or key.rsplit("/", 1)[-1] == tail]
        return matches[0] if len(matches) == 1 else None

    def cost_for(self, model: str, usage: Usage) -> tuple[Decimal, bool]:
        """(비용, 가격을 알고 있었는가) 를 돌려줍니다."""
        price = self._lookup(model)
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
