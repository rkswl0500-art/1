"""비용/토큰/지연 원장.

슬라이스 1 은 CostMeter(사후 기록)만 씁니다. Estimator(사전 견적)는
슬라이스 3 에서 이 파일에 붙습니다.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from decimal import Decimal

from .config import PricingTable
from .models import CallRecord, Usage

_HANGUL = (
    (0xAC00, 0xD7A3),  # 완성형 음절
    (0x1100, 0x11FF),  # 자모
    (0x3130, 0x318F),  # 호환 자모
)


def _is_hangul(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _HANGUL)


def estimate_tokens(text: str, ko_tokens_per_char: float = 1.0) -> int:
    """토큰 수 근사.

    한글은 모델별 토크나이저 편차가 커서 영어의 "4자=1토큰" 규칙이 안 통합니다.
    한글 문자에는 설정 계수를, 나머지에는 0.25 를 적용하는 조잡한 근사이고,
    **견적 전용**입니다. 기록되는 실사용량은 언제나 API 응답의 usage 이므로
    이 함수가 틀려도 원장은 정확합니다. 계수는 슬라이스 3 에서 실측 보정합니다.
    """
    if not text:
        return 0
    ko = sum(1 for ch in text if _is_hangul(ch))
    return max(1, int(ko * ko_tokens_per_char + (len(text) - ko) * 0.25))


@dataclass(frozen=True, slots=True)
class CostReport:
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_usd: Decimal
    by_model: dict[str, Decimal]
    unpriced_models: tuple[str, ...]
    latencies_ms: tuple[int, ...]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostMeter:
    """모든 LLM 호출의 단일 원장.

    용도(토론/쟁점/요약/판정)를 불문하고 한 곳에 모입니다. 슬라이스 3 에서
    Storage 가 이 레코드를 그대로 sqlite 에 밀어넣습니다.
    """

    def __init__(self, pricing: PricingTable, debate_id: str) -> None:
        self._pricing = pricing
        self._debate_id = debate_id
        self._records: list[CallRecord] = []
        self._ids = itertools.count(1)

    def record(
        self,
        *,
        purpose: str,
        agent_id: str | None,
        provider: str,
        model: str,
        usage: Usage,
        latency_ms: int,
        attempts: int = 1,
        finish_reason: str | None = None,
    ) -> CallRecord:
        cost, priced = self._pricing.cost_for(model, usage)
        rec = CallRecord(
            call_id=f"{self._debate_id}-c{next(self._ids)}",
            debate_id=self._debate_id,
            purpose=purpose,  # type: ignore[arg-type]
            agent_id=agent_id,
            provider=provider,
            model=model,
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=cost,
            priced=priced,
            attempts=attempts,
            finish_reason=finish_reason,
        )
        self._records.append(rec)
        return rec

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(self._records)

    def report(self) -> CostReport:
        by_model: dict[str, Decimal] = {}
        for r in self._records:
            by_model[r.model] = by_model.get(r.model, Decimal(0)) + r.cost_usd
        return CostReport(
            calls=len(self._records),
            prompt_tokens=sum(r.usage.prompt_tokens for r in self._records),
            completion_tokens=sum(r.usage.completion_tokens for r in self._records),
            total_usd=sum((r.cost_usd for r in self._records), Decimal(0)),
            by_model=by_model,
            unpriced_models=self._pricing.unpriced_models,
            latencies_ms=tuple(r.latency_ms for r in self._records),
        )
