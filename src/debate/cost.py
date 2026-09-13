"""비용/토큰/지연 원장과 사전 견적."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from decimal import Decimal

from .config import PricingTable
from typing import Sequence

from .judge import verdict_content_tokens
from .models import CallRecord, Usage, build_header

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
    #: 판정이 한 번 실패해 복구 호출이 붙었을 때 추가되는 몫.
    #:
    #: 본 범위에 섞지 않습니다. 섞으면 범위가 1.7배로 벌어져서 "보통 얼마"인지를
    #: 잃습니다. 복구는 일어나거나 안 일어나거나이므로 따로 보여주는 편이
    #: 판단에 쓰입니다. 실측: 복구 없는 실행 6,777 / 복구 1회 실행 9,239.
    judge_repair_tokens: int = 0
    judge_repair_usd: Decimal = Decimal(0)

    def format(self) -> str:
        lines = [
            f"호출 {self.calls}회 | 토큰 {self.tokens_low:,} ~ {self.tokens_high:,}",
            f"예상 비용: ${self.low_usd.quantize(Decimal('0.0001'))} ~ "
            f"${self.high_usd.quantize(Decimal('0.0001'))}",
        ]
        for model, (lo, hi) in sorted(self.per_model.items()):
            lines.append(f"  {model}: ${lo.quantize(Decimal('0.0001'))} ~ "
                         f"${hi.quantize(Decimal('0.0001'))}")
        if self.judge_repair_tokens:
            lines.append(
                f"  판정 복구가 붙으면 +{self.judge_repair_tokens:,} tok "
                f"(+${self.judge_repair_usd.quantize(Decimal('0.0001'))}) — "
                f"심판 출력이 한 번에 파싱되지 않으면 1회 더 호출합니다")
        if self.unpriced_models:
            lines.append(f"  가격 미상: {', '.join(self.unpriced_models)} — 0 으로 계상")
        lines.append(f"  (한국어 토큰 계수 {self.ko_tokens_per_char}, "
                     f"실행 후 실측과 비교해 보정하십시오)")
        return "\n".join(lines)


#: 발언 1건의 출력 길이 가정. 라운드 성격에 따라 다릅니다.
#:
#: R1 은 입론이라 짧고(실측 250~450자), R2 이후는 반박이라 구조가 더 붙습니다
#: (인용 → 문제점 → 근거 + 입장 유지 설명). 하나의 넓은 범위로 뭉개면 정직한 게
#: 아니라 쓸모없어집니다 — 규칙 상한은 600자입니다.
_OUT_CHARS_R1 = (250, 450)
_OUT_CHARS_REBUTTAL = (350, 600)
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
        moderator_model: str | None = None, topic: str = "",
        issue_count: int = 4,
    ) -> CostEstimate:
        n = len(participants)
        per_model_lo: dict[str, Decimal] = {}
        per_model_hi: dict[str, Decimal] = {}
        tok_lo = tok_hi = calls = 0

        def add(model: str, in_lo: int, in_hi: int, out_lo: int, out_hi: int) -> None:
            """입력도 저/고를 따로 받습니다.

            입력에 단일 값을 쓰면 저 추정에도 **직전 라운드가 최대로 길었을 때**의
            크기가 들어가, 하한이 하한이 아니게 됩니다. 3라운드 실측(7,307)이
            하한(7,713) 아래로 떨어진 원인이 이것이었습니다.
            """
            nonlocal tok_lo, tok_hi, calls
            calls += 1
            tok_lo += in_lo + out_lo
            tok_hi += in_hi + out_hi
            lo, _ = self._pricing.cost_for(model, Usage(in_lo, out_lo))
            hi, _ = self._pricing.cost_for(model, Usage(in_hi, out_hi))
            per_model_lo[model] = per_model_lo.get(model, Decimal(0)) + lo
            per_model_hi[model] = per_model_hi.get(model, Decimal(0)) + hi

        transcript_lo: list[int] = []   # 라운드별 발언 총량(저 추정), Judge 용
        transcript_hi: list[int] = []

        # 헤더는 참가자마다 실제로 만들어 잽니다. 입장·페르소나를 넣으면 길어지고,
        # 고정값으로 두면 그 차이를 영영 반영하지 못합니다.
        header_chars = {
            spec.model: len(build_header(spec, topic)) for spec in participants
        }

        for round_no in range(1, rounds + 1):
            rebuttal = round_no > 1
            out_lo, out_hi = _OUT_CHARS_REBUTTAL if rebuttal else _OUT_CHARS_R1
            # 직전 라운드 전문은 (참가자 수 - 1) 명분이 들어갑니다. 저/고를
            # 각각 직전 라운드의 저/고 출력으로 잡습니다.
            prev = _OUT_CHARS_R1 if round_no == 2 else _OUT_CHARS_REBUTTAL
            prev_lo = 0 if not rebuttal else (n - 1) * prev[0]
            prev_hi = 0 if not rebuttal else (n - 1) * prev[1]
            extras = 0 if not rebuttal else _ISSUES_CHARS + _DIGEST_CHARS
            for spec in participants:
                base = header_chars[spec.model] + extras
                add(spec.model,
                    self._tok(base + prev_lo), self._tok(base + prev_hi),
                    self._tok(out_lo), self._tok(out_hi))
            transcript_lo.append(n * out_lo)
            transcript_hi.append(n * out_hi)

        mod = moderator_model or (participants[0].model if participants else None)
        if mod and rounds >= 1:
            # 쟁점 추출 1회 + 요약 (rounds - 2) 회
            add(mod, self._tok(n * _OUT_CHARS_R1[0]), self._tok(n * _OUT_CHARS_R1[1]),
                self._tok(100), self._tok(_ISSUES_CHARS))
            for _ in range(max(0, rounds - 2)):
                add(mod,
                    self._tok(n * _OUT_CHARS_REBUTTAL[0] + _DIGEST_CHARS),
                    self._tok(n * _OUT_CHARS_REBUTTAL[1] + _DIGEST_CHARS),
                    self._tok(150), self._tok(_DIGEST_CHARS))

        repair_tokens, repair_usd = 0, Decimal(0)
        if judge_model:
            # Judge 는 전 라운드 전문을 한 번에 봅니다 — 단일 패스이므로.
            before_hi, before_usd = tok_hi, sum(per_model_hi.values(), Decimal(0))
            # 판정 출력은 쟁점·참가자·루브릭 축 수에 비례합니다. judge.py 가
            # 예산을 잡을 때 쓰는 함수를 그대로 씁니다 — 따로 추정하면 한쪽만
            # 갱신되어 어긋납니다(실제로 견적 쪽이 쟁점 수를 무시하고 있었습니다).
            content = verdict_content_tokens(issue_count, n)
            add(judge_model,
                self._tok(sum(transcript_lo) + _ISSUES_CHARS),
                self._tok(sum(transcript_hi) + _ISSUES_CHARS),
                content // 2, content)
            # 복구 호출은 같은 프롬프트를 다시 보내는 것이라 한 번 더 친 것과
            # 비슷합니다. 본 범위가 아니라 별도 항목으로 냅니다.
            repair_tokens = tok_hi - before_hi
            repair_usd = sum(per_model_hi.values(), Decimal(0)) - before_usd

        return CostEstimate(
            low_usd=sum(per_model_lo.values(), Decimal(0)),
            high_usd=sum(per_model_hi.values(), Decimal(0)),
            tokens_low=tok_lo, tokens_high=tok_hi, calls=calls,
            per_model={m: (per_model_lo[m], per_model_hi[m]) for m in per_model_lo},
            unpriced_models=self._pricing.unpriced_models,
            ko_tokens_per_char=self._ko,
            judge_repair_tokens=repair_tokens,
            judge_repair_usd=repair_usd,
        )
