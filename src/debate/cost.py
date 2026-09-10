"""비용/토큰/지연 원장과 사전 견적."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from decimal import Decimal

from .config import PricingTable
from typing import Sequence

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
        round_no: int | None = None,
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
            round_no=round_no,
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


# ── 사전 견적 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CostEstimate:
    low_usd: Decimal
    high_usd: Decimal
    tokens_low: int
    tokens_high: int
    calls: int
    per_model: dict[str, tuple[Decimal, Decimal]]
    unpriced_models: tuple[str, ...]
    ko_tokens_per_char: float

    def format(self) -> str:
        lines = [
            f"호출 {self.calls}회 | 토큰 {self.tokens_low:,} ~ {self.tokens_high:,}",
            f"예상 비용: ${self.low_usd.quantize(Decimal('0.0001'))} ~ "
            f"${self.high_usd.quantize(Decimal('0.0001'))}",
        ]
        for model, (lo, hi) in sorted(self.per_model.items()):
            lines.append(f"  {model}: ${lo.quantize(Decimal('0.0001'))} ~ "
                         f"${hi.quantize(Decimal('0.0001'))}")
        if self.unpriced_models:
            lines.append(f"  가격 미상: {', '.join(self.unpriced_models)} — 0 으로 계상")
        lines.append(f"  (한국어 토큰 계수 {self.ko_tokens_per_char}, "
                     f"실행 후 실측과 비교해 보정하십시오)")
        return "\n".join(lines)


#: 발언 1건의 출력 길이 가정. 규칙이 600자 이내이므로 그 범위를 폭으로 씁니다.
_OUT_CHARS_LOW, _OUT_CHARS_HIGH = 200, 600
#: 고정 헤더(주제 + 페르소나 + 규칙)의 대략적 크기.
_HEADER_CHARS = 400
#: 쟁점 목록과 요약본의 상한. 둘 다 길이가 묶여 있습니다.
_ISSUES_CHARS, _DIGEST_CHARS = 300, 400


class Estimator:
    """LLM 을 한 번도 부르지 않고 비용 범위를 계산합니다.

    **범위로 냅니다.** 출력 길이를 모르는데 단일 숫자를 내면 그건 거짓말입니다.

    컨텍스트가 라운드마다 달라지는 걸 반영합니다 — R1 은 헤더뿐이지만 R2 부터는
    쟁점 + 직전 라운드 전문(참가자 수에 비례) + 요약이 붙습니다. 참가자를 늘리면
    라운드당 호출 수와 각 호출의 입력이 **함께** 커져서 비용이 대략 N² 로
    움직입니다. 이걸 모델링하지 않으면 5명에서 크게 과소 예측합니다.
    """

    def __init__(self, pricing: PricingTable, ko_tokens_per_char: float = 1.0) -> None:
        self._pricing = pricing
        self._ko = ko_tokens_per_char

    def _tok(self, chars: int) -> int:
        return max(1, int(chars * self._ko))

    def estimate(
        self, *, participants: Sequence, rounds: int, judge_model: str | None,
        moderator_model: str | None = None,
    ) -> CostEstimate:
        n = len(participants)
        per_model_lo: dict[str, Decimal] = {}
        per_model_hi: dict[str, Decimal] = {}
        tok_lo = tok_hi = calls = 0

        def add(model: str, in_tok: int, out_lo: int, out_hi: int) -> None:
            nonlocal tok_lo, tok_hi, calls
            calls += 1
            tok_lo += in_tok + out_lo
            tok_hi += in_tok + out_hi
            lo, _ = self._pricing.cost_for(model, Usage(in_tok, out_lo))
            hi, _ = self._pricing.cost_for(model, Usage(in_tok, out_hi))
            per_model_lo[model] = per_model_lo.get(model, Decimal(0)) + lo
            per_model_hi[model] = per_model_hi.get(model, Decimal(0)) + hi

        transcript_lo: list[int] = []   # 라운드별 발언 총량(저 추정), Judge 용
        transcript_hi: list[int] = []

        for round_no in range(1, rounds + 1):
            # 직전 라운드 전문은 (참가자 수 - 1) 명분이 들어갑니다.
            prev_chars = 0 if round_no == 1 else (n - 1) * _OUT_CHARS_HIGH
            extras = 0 if round_no == 1 else _ISSUES_CHARS + _DIGEST_CHARS
            in_tok = self._tok(_HEADER_CHARS + prev_chars + extras)
            for spec in participants:
                add(spec.model, in_tok,
                    self._tok(_OUT_CHARS_LOW), self._tok(_OUT_CHARS_HIGH))
            transcript_lo.append(n * _OUT_CHARS_LOW)
            transcript_hi.append(n * _OUT_CHARS_HIGH)

        mod = moderator_model or (participants[0].model if participants else None)
        if mod and rounds >= 1:
            # 쟁점 추출 1회 + 요약 (rounds - 2) 회
            add(mod, self._tok(n * _OUT_CHARS_HIGH), self._tok(100), self._tok(_ISSUES_CHARS))
            for _ in range(max(0, rounds - 2)):
                add(mod, self._tok(n * _OUT_CHARS_HIGH + _DIGEST_CHARS),
                    self._tok(150), self._tok(_DIGEST_CHARS))

        if judge_model:
            # Judge 는 전 라운드 전문을 한 번에 봅니다 — 단일 패스이므로.
            add(judge_model, self._tok(sum(transcript_hi) + _ISSUES_CHARS),
                self._tok(300), self._tok(1200))

        return CostEstimate(
            low_usd=sum(per_model_lo.values(), Decimal(0)),
            high_usd=sum(per_model_hi.values(), Decimal(0)),
            tokens_low=tok_lo, tokens_high=tok_hi, calls=calls,
            per_model={m: (per_model_lo[m], per_model_hi[m]) for m in per_model_lo},
            unpriced_models=self._pricing.unpriced_models,
            ko_tokens_per_char=self._ko,
        )
