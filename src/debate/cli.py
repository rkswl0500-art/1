"""터미널 진입점. 슬라이스 1 의 검증 표면입니다.

    python -m debate.cli models --provider fake
    python -m debate.cli run --provider-override fake \
        --topic "..." --agent fake-a --agent fake-b --rounds 1
"""

from __future__ import annotations

import argparse
import asyncio
import time
import itertools
import json
import string
import sys
from decimal import Decimal
from pathlib import Path

import yaml

from .agent import (
    Agent, Anonymizer, Moderator, parse_participant_line, split_provider_model,
)
from .config import (
    FAKE_PROVIDER, PricingTable, Settings, load_provider_slots, provider_registry,
)
from .cost import CostMeter, Estimator
from .context import ContextBuilder
from .engine import DebateEngine
from .judge import Judge, family_note, score_contradictions, truncation_advice
from .session import new_debate_id
from .storage import SqliteStore
from .models import (
    AgentDropped, AgentSpec, ConfigError, DebateConfig, DebateResult,
    IssuesExtracted, RoundStarted, RoundSummarized, UtteranceCompleted,
    UtteranceStarted,
)
from .provider import (
    FakeBehavior, FakeProvider, ProviderError, build_pool, looks_non_chat,
)

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
    """`provider/model [| 입장 [| 페르소나]]` 를 AgentSpec 으로.

    구분자가 '|' 인 이유는 모델 ID 에 콜론이 들어가기 때문입니다
    (OpenRouter 무료 모델의 `:free` 접미사).
    """
    head, stance, persona = parse_participant_line(raw)

    if "/" in head:
        provider, _, model = head.partition("/")
    elif fake:
        provider, model = FAKE_PROVIDER, head
    else:
        raise ConfigError(
            f"--agent {raw!r}: 'provider/model' 형식으로 쓰세요 "
            "(예: openai/gpt-4o-mini, 또는 'openai/gpt-4o-mini | 반대 | 회의주의자'). "
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
        persona=persona or DEFAULT_PERSONA,
        stance=stance,
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


def _build_fake(
    specs: list[AgentSpec], latencies: str | None,
    fail_agent: str | None = None, fail_mode: str = "500",
) -> FakeProvider:
    """--fake-latency 1800,2100 과 --fail-agent m2@round2 를 행동표로 만듭니다."""
    values: list[int] = []
    if latencies:
        values = [int(v) for v in latencies.split(",") if v.strip()]
        if len(values) != len(specs):
            raise ConfigError(
                f"--fake-latency 값이 {len(values)}개인데 참가자는 {len(specs)}명입니다"
            )

    rounds: frozenset[int] = frozenset()
    target = fail_agent
    if fail_agent and "@" in fail_agent:
        target, _, spec = fail_agent.partition("@")
        digits = "".join(ch for ch in spec if ch.isdigit())
        if not digits:
            raise ConfigError(
                f"--fail-agent {fail_agent!r}: '모델@round2' 형식으로 쓰세요"
            )
        rounds = frozenset({int(digits)})

    if target and not any(s.model == target or s.id == target for s in specs):
        known = ", ".join(f"{s.id}={s.model}" for s in specs)
        raise ConfigError(f"--fail-agent {target!r} 가 참가자에 없습니다. 참가자: {known}")

    behaviors: dict[str, FakeBehavior] = {}
    for i, spec in enumerate(specs):
        hit = target is not None and (spec.model == target or spec.id == target)
        behaviors[spec.model] = FakeBehavior(
            latency_ms=values[i] if values else 800,
            fail_mode=fail_mode,  # type: ignore[arg-type]
            fail_always=hit and not rounds,
            fail_rounds=rounds if hit else frozenset(),
        )
    return FakeProvider(behaviors)


# ── 출력 ─────────────────────────────────────────────────────────────────────


def _n(x: int) -> str:
    return f"{x:,}"


def _usd(x: Decimal) -> str:
    return f"${x.quantize(Decimal('0.0001'))}"


_current_budget = "미상"


class _Progress:
    """라운드가 도는 동안 터미널에 진행 상황을 흘립니다.

    느린 모델이 섞이면(관측: 54.3s vs 1.8s, 30배) 라운드 전체가 가장 느린
    참가자에 묶입니다. 진행 표시가 없으면 그 시간이 통째로 무응답으로 보여서
    멈춘 건지 기다리는 건지 알 수 없습니다. 슬라이스 4 의 SSE 가 쓸 이벤트
    싱크를 그대로 사용하므로, 여기서 배선이 검증됩니다.
    """

    def __init__(self, labels: dict[str, str], concurrency: int) -> None:
        self._labels = labels
        self._concurrency = concurrency
        self._done = 0
        self._active = 0

    async def __call__(self, event) -> None:
        if isinstance(event, RoundStarted):
            self._done, self._active = 0, len(event.active)
            names = ", ".join(self._labels[a] for a in event.active)
            print(f"\n[R{event.round_no}] 시작 — {names} (동시 {self._concurrency})",
                  flush=True)
        elif isinstance(event, UtteranceCompleted):
            self._done += 1
            u = event.utterance
            print(f"  {event.label} 완료 ({_n(u.latency_ms)}ms, "
                  f"{u.usage.completion_tokens}tok)  [{self._done}/{self._active}]",
                  flush=True)
        elif isinstance(event, UtteranceStarted):
            print(f"  {event.label} 말하는 중…", flush=True)
        elif isinstance(event, IssuesExtracted):
            print(f"\n쟁점 ({len(event.issues)}):", flush=True)
            for n, issue in enumerate(event.issues, start=1):
                print(f"  {n}) {issue.title}", flush=True)
        elif isinstance(event, RoundSummarized):
            print(f"  [R{event.round_no} 요약 접힘, {len(event.digest)}자]", flush=True)
        elif isinstance(event, AgentDropped):
            print(f"  {self._labels.get(event.agent_id, event.agent_id)} 드롭 — "
                  f"{event.reason}", flush=True)


def _make_dump(dump_dir: str | None):
    """원본 응답을 파일로 남깁니다. 잘림 원인 규명용.

    요청 헤더는 저장하지 않습니다 — Authorization 이 그대로 파일에 남습니다.
    저장하는 것은 보낸 메시지와 받은 JSON 본문뿐입니다.
    """
    if not dump_dir:
        return None
    out = Path(dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    counter = itertools.count(1)

    def dump(req, body: dict) -> None:
        path = out / f"{next(counter):02d}-{req.model.replace('/', '_')}.json"
        path.write_text(
            json.dumps(
                {
                    "request": {
                        "model": req.model,
                        "max_tokens": req.max_tokens,
                        "temperature": req.temperature,
                        "messages": [{"role": m.role, "content": m.content}
                                     for m in req.messages],
                    },
                    "response": body,
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[dump] {path}")

    return dump


def _preview_context(builder, specs, cfg) -> None:
    """가짜 상태로 마지막 라운드 컨텍스트를 미리 찍어 눈으로 검사하게 합니다.

    라운드 내 블라인드는 눈으로 확인할 수 있어야 믿을 수 있습니다 — 테스트는
    불변식을 지키지만, 프롬프트가 실제로 어떻게 보이는지는 사람이 봐야 합니다.
    """
    from .models import AnonUtterance, DebateState, Issue, RoundResult, Usage, Utterance

    state = DebateState(topic=cfg.topic, round_no=cfg.rounds)
    state.issues = tuple(Issue(f"i{n}", f"<쟁점 {n} 자리>") for n in range(1, 4))
    state.prior_digest = "<R1..R(n-2) 요약 자리>"
    state.completed_rounds = [RoundResult(
        cfg.rounds - 1,
        tuple(Utterance(s.id, cfg.rounds - 1, f"<{s.label} 의 직전 라운드 발언>",
                        Usage(0, 0), 0, Decimal(0)) for s in specs),
        0, 0, 1,
    )]
    pack = builder.build_for(specs[0], state)
    print(f"\n{'='*70}\n{specs[0].label} 에게 갈 R{cfg.rounds} 컨텍스트 (미리보기)\n{'='*70}")
    for msg in pack.render():
        print(f"--- {msg.role} ---")
        print(msg.content)
    print("=" * 70)


def _resolve_judge(args, specs: list[AgentSpec], settings) -> tuple[str, str] | None:
    """(provider, model). Judge 를 참가자 풀에서 배제합니다.

    검증은 **모델 ID 비교**입니다. 같은 벤더 계열(예: gemini-flash 참가자 +
    gemini-pro judge)은 이 검사를 통과하면서도 자기편애가 남습니다 — 계열 기준이
    애매해서 막지 않기로 한 결정이고, 대신 실행 끝에 계열을 사실로 찍습니다.
    """
    if not args.judge:
        return None
    provider, model = split_provider_model(args.judge, specs[0].provider)
    if not model:
        raise ConfigError(f"--judge {args.judge!r}: 'provider/model' 형식으로 쓰세요")

    overlap = sorted({s.model for s in specs if s.model == model})
    if overlap and not args.allow_judge_overlap:
        raise ConfigError(
            f"judge 모델 {model!r} 이 참가자 풀에 있습니다. "
            "LLM 은 자기 출력을 편애해서 점수가 오염됩니다. 다른 모델을 쓰거나, "
            "모델이 부족하면 --allow-judge-overlap 으로 명시적으로 허용하세요."
        )
    return provider, model


def _print_verdict(verdict) -> None:
    print(f"\n{'─' * 60}\n판정 (judge: {verdict.judge_model})")
    if verdict.status == "unparsed":
        # "파싱 실패"로만 적으면 형식 문제로 오해합니다. 잘린 것은 예산 문제라
        # 대처가 완전히 다릅니다.
        if verdict.failure_kind == "truncated":
            print("  ⚠ 판정이 길이 제한에 걸려 중간에 잘렸습니다 "
                  "(finish_reason=length). 형식 문제가 아닙니다.")
            print(f"     {truncation_advice(verdict)} 시도 {verdict.parse_attempts}회.")
        else:
            print("  ⚠ 판정이 올바른 JSON 이 아니었습니다 (형식 오류). "
                  f"시도 {verdict.parse_attempts}회.")
        print(f"  원문 {len(verdict.raw)}자, verdicts.raw 에 저장했습니다.")
        print(f"  앞부분: {verdict.raw[:160]!r}")
        return

    for score in verdict.per_issue:
        pairs = "  ".join(f"{k} {v}" for k, v in sorted(score.scores.items()))
        print(f"  [{score.issue_id}] {pairs}")
        if score.reasoning:
            print(f"        {score.reasoning}")
    axes = sorted({k for scores in verdict.rubric.values() for k in scores})
    for label in sorted(verdict.rubric):
        row = verdict.rubric[label]
        print(f"  {label}: " + "  ".join(f"{k} {row.get(k, '-')}" for k in axes))
    totals = verdict.totals()
    print("  합계: " + "  ".join(f"{k} {v}" for k, v in sorted(totals.items())))
    winner = verdict.winner or "무승부"
    print(f"  승자: {winner} ({verdict.margin})")
    if verdict.conclusion:
        print(f"  결론: {verdict.conclusion}")
    if verdict.dissent:
        print(f"  미해결 반론: {verdict.dissent}")
    if verdict.truncated:
        print("  ⚠ 판정이 잘렸습니다(finish_reason=length). "
              "단일 패스 한계에 도달했을 수 있습니다.")
    if verdict.repaired_from_truncation:
        # status='ok' / finish_reason='stop' 이라 기록만 보면 아무 일도 없던
        # 것처럼 보이지만, 복구 프롬프트가 "짧게 쓰라"고 지시했습니다.
        print("  ⚠ 1차 판정이 잘려 **더 짧게 다시 받았습니다**. 점수는 유효하지만 "
              "이 심판이 평소 쓰는 길이는 아닙니다.")
        if verdict.first_budget and verdict.first_output_tokens:
            gap = verdict.first_budget - verdict.first_output_tokens
            print(f"     1차: 본문 {verdict.first_output_tokens:,}tok / 예산 "
                  f"{verdict.first_budget:,}tok — 차액 {gap:,}tok 은 응답에 "
                  "안 보이는 사고 토큰으로 보입니다.")
        if (verdict.repair_reasoning_effort
                and verdict.repair_reasoning_effort != verdict.reasoning_effort):
            print(f"     복구 호출은 사고 수준을 {verdict.reasoning_effort} → "
                  f"{verdict.repair_reasoning_effort} 로 낮췄습니다.")
    print(f"  사고 수준: {verdict.reasoning_effort or '모델 기본값(보내지 않음)'}")
    if verdict.reasoning_note:
        print(f"  ⚠ 사고 수준을 바꿔 보냈습니다: {verdict.reasoning_note}")
    for note in score_contradictions(verdict):
        # 점수 필드는 정확합니다. 산문만 틀렸으므로 표를 믿으라고 알려줍니다.
        print(f"  ⚠ {note}\n     위 표의 점수가 정확합니다.")


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
            flag = "  ⚠ 잘림(finish_reason=length)" if u.truncated else ""
            print(f"{head}  ({_n(u.latency_ms)}ms, {models[u.agent_id]}, "
                  f"{u.usage.completion_tokens}tok){flag}")
            for line in u.content.splitlines() or [""]:
                print(f"  {line}")
            print(f"  └ {len(u.content)}자 / finish_reason={u.finish_reason!r}")
            print()

        print(
            f"wall {_n(rnd.wall_ms)}ms | "
            f"sum(ok={rnd.ok_count}) {_n(rnd.sum_latency_ms)}ms | "
            f"max {_n(rnd.max_latency_ms)}ms | "
            f"failed={rnd.failed_count} | waves {rnd.waves}"
        )

    if result.issues:
        print(f"\n확정된 쟁점 {len(result.issues)}개: "
              + " | ".join(i.title for i in result.issues))
    if result.dropped:
        print(f"dropout: {', '.join(result.dropped)}")
    for warning in result.warnings:
        print(f"경고: {warning}")
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

    _print_diagnosis(result, rep)


def _print_diagnosis(result: DebateResult, rep) -> None:
    """라이브 스모크에서 헛다리를 짚지 않게 하는 힌트.

    두 경우가 조용히 지나가기 쉬워서 명시적으로 짚어줍니다:
      - FatalError 는 재시도로 안 풀리는 설정 문제인데, 참가자 단위로 격리되는
        바람에 헤드라인이 "참가자 부족"으로 보입니다. 실제 원인은 키/모델 ID 입니다.
      - usage 를 안 돌려주는 엔드포인트는 토큰이 0 으로 집계되는데 종료 코드는
        0 입니다. 비용 추적이 조용히 죽는 유일한 경로입니다.
    """
    fatal = [
        u for r in result.rounds for u in r.utterances
        if u.status == "failed" and "FatalError" in (u.error or "")
    ]
    if fatal:
        print(
            "\n힌트: FatalError 는 재시도로 해결되지 않는 설정 문제입니다.\n"
            "      401/403 → API_KEY, 404 → 모델 ID 오타 또는 BASE_URL 경로.\n"
            "      .env 의 DEBATE_PROVIDER_*_{API_KEY,BASE_URL} 과 모델 ID 를 확인하세요."
        )

    truncated = [
        u for r in result.rounds for u in r.utterances
        if u.status == "ok" and u.truncated
    ]
    if truncated:
        who = ", ".join(sorted({u.agent_id for u in truncated}))
        print(
            f"\n경고: 발언이 잘렸습니다 (finish_reason=length): {who}\n"
            f"      출력 토큰 예산이 부족합니다. DEBATE_MAX_OUTPUT_TOKENS 를 올리세요\n"
            f"      (현재 {_current_budget}). 잘린 발언은 Judge 채점을 그대로 오염시킵니다."
        )

    if rep.calls and rep.total_tokens == 0:
        print(
            "\n힌트: 호출은 성공했는데 토큰이 0 입니다. 이 엔드포인트가 응답에 usage 를\n"
            "      담지 않는 것으로 보입니다. 비용/토큰 집계가 전부 0 이 되므로\n"
            "      슬라이스 3 의 견적·원장을 신뢰할 수 없습니다. 보고해 주세요."
        )


# ── 커맨드 ───────────────────────────────────────────────────────────────────


async def _cmd_models(args: argparse.Namespace) -> int:
    settings = Settings()
    name = args.provider.lower()
    def show(models: list[str]) -> None:
        for m in models:
            reason = looks_non_chat(m)
            print(f"{m}    ← {reason} (채팅 불가로 보임)" if reason else m)

    if name == FAKE_PROVIDER:
        show(await FakeProvider().list_models())
        return 0

    registry = provider_registry(settings)
    if name not in registry:
        raise ConfigError(
            f"프로바이더 {name!r} 를 찾을 수 없습니다.\n  {registry.source_hint()}"
        )
    from .provider import OpenAICompatProvider

    provider = OpenAICompatProvider(
        registry.get(name), timeout_s=settings.request_timeout_s
    )
    try:
        show(await provider.list_models())
    finally:
        await provider.aclose()
    return 0


def _model_key(model: str) -> str:
    """`models/gemini-3.5-flash-lite` 와 `gemini-3.5-flash-lite` 를 한 행으로.

    같은 모델이 요청과 응답에서 접두사가 다르게 옵니다. 그대로 묶으면 통계가
    둘로 갈려서 건수가 적어 보이고 평균도 따로 놉니다. `PricingTable._lookup`
    과 같은 규칙(마지막 구간)입니다.
    """
    return model.rsplit("/", 1)[-1]


def _judge_key(model: str, first_effort: str | None) -> str:
    """판정 집계 키: 모델 + **그 토론 1차 호출의** 사고 수준.

    사고 수준이 다르면 같은 모델이라도 다른 측정입니다 — low 로 돈 판정과
    모델 기본값(gemini-3-flash-preview 는 high)으로 돈 판정을 한 평균에 섞으면
    둘 다 의미를 잃습니다. NULL 은 '보내지 않음' 이고, 이 컬럼이 생기기 전의
    기록도 전부 여기에 들어갑니다(그때는 보낸 적이 없으므로 맞습니다).

    복구 호출은 사고 수준을 한 단계 낮춰 보내므로 **자기 값**으로 묶으면 1차가
    0건인 유령 행이 생깁니다. 그래서 1차의 값을 씁니다.
    """
    return f"{_model_key(model)} [{first_effort or '기본값'}]"


#: 잘리지 않고 정상 종료한 것으로 보는 값. 그 외에는 전부 세어서 보여줍니다 —
#: 프로바이더마다 문자열이 달라서(`length`, `MAX_TOKENS`, …) 목록을 미리
#: 고정하면 모르는 값이 조용히 '정상' 으로 들어갑니다.
_CLEAN_FINISH = {"stop", "end_turn", "eos", None, ""}


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """기록된 실사용량을 견적 모델과 비교합니다.

    "Groq 심판이 Gemini 보다 길게 쓰는가" 같은 질문을 **추측이 아니라 데이터로**
    답하려고 있습니다. 벤더별 상수를 넣을지 말지는 여기 숫자가 쌓인 뒤에
    결정할 문제입니다 — 한 건으로 배수를 확정하면 보정이 아니라 과적합입니다.

    **fake 실행은 뺍니다.** fake 모드로 돌려도 모델명은 사용자가 고른 실제
    이름이 그대로 남습니다(그 설정을 리허설했다는 기록이라 지우지 않습니다).
    `provider` 가 `fake` 라 구분은 명확한데, 예전 쿼리가 모델명으로만 묶고
    provider 를 안 봐서 fake 출력이 실제 모델의 실측치로 보고됐습니다.
    뺀 건수는 **반드시 찍습니다** — 조용히 거르면 집계가 왜 비었는지 모릅니다.
    """
    import sqlite3

    from .judge import (
        required_output_tokens, thinking_reserve_for, verdict_content_tokens,
    )

    settings = Settings()
    if not settings.db_path.is_file():
        print(f"기록이 없습니다: {settings.db_path}", file=sys.stderr)
        return 2

    # 스키마를 먼저 맞춥니다. pull 직후 토론을 한 번도 안 돌리고 calibrate 부터
    # 돌리면 옛 DB 에는 reasoning_effort 컬럼이 없어 쿼리가 죽습니다.
    SqliteStore(settings.db_path).close()
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row

    excluded = conn.execute(
        "SELECT COUNT(*) FROM llm_calls WHERE provider = ?", (FAKE_PROVIDER,)
    ).fetchone()[0]
    if excluded:
        print(f"fake 실행 {excluded}건을 뺐습니다 (provider='{FAKE_PROVIDER}').")
        print("  fake 는 모델명이 고른 그대로 남으므로, 모델명으로만 묶으면 "
              "실제 모델의 실측치로 보입니다.\n")

    # 한 토론에 판정 호출이 둘 이상이면 뒤엣것은 **복구 호출**입니다. 복구
    # 프롬프트는 "각 reasoning 을 한 문장으로" 라고 짧게 쓰라 지시하므로, 그
    # 응답 길이는 이 모델이 평소 쓰는 길이가 아닙니다. 평균에 섞으면 짧은 쪽으로
    # 치우칩니다 — 잘린 건을 빼는 것만으로는 부족합니다.
    rows = list(conn.execute("""
        SELECT l.model, l.out_tok, l.finish_reason, d.debate_id,
               ROW_NUMBER() OVER (PARTITION BY l.debate_id ORDER BY l.rowid) AS seq,
               FIRST_VALUE(l.reasoning_effort) OVER (
                   PARTITION BY l.debate_id ORDER BY l.rowid) AS first_effort,
               (SELECT COUNT(*) FROM issues i WHERE i.debate_id = d.debate_id) AS issues,
               (SELECT COUNT(*) FROM participants p WHERE p.debate_id = d.debate_id) AS parts
        FROM llm_calls l JOIN debates d USING (debate_id)
        WHERE l.purpose = 'judge' AND l.out_tok > 0 AND l.provider IS NOT ?
        ORDER BY l.model
    """, (FAKE_PROVIDER,)))
    if not rows:
        print("판정 호출 기록이 없습니다 (fake 제외).")
        return 0

    print("판정 출력: 실측 대 모델 (verdict_content_tokens)\n")
    print(f"  {'심판 [사고 수준]':36} {'1차':>4} {'잘림':>4} {'복구':>4} {'실측 평균':>9} "
          f"{'모델 평균':>9} {'비율':>6}")
    print("  " + "─" * 82)

    by_model: dict[str, list[tuple[int, int, bool, bool]]] = {}
    spellings: dict[str, set[str]] = {}
    finishes: dict[str, int] = {}
    #: 잘린 1차 호출의 (본문 토큰, 그때 준 예산). 기록에는 예산이 없지만 자동
    #: 계산이었다면 쟁점·참가자 수로 다시 구할 수 있습니다.
    cut_vs_budget: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        key = _judge_key(r["model"], r["first_effort"])
        spellings.setdefault(_model_key(r["model"]), set()).add(r["model"])
        cut = r["finish_reason"] not in _CLEAN_FINISH
        repair = r["seq"] > 1
        if cut:
            finishes[r["finish_reason"]] = finishes.get(r["finish_reason"], 0) + 1
            if not repair:
                budget = settings.judge_max_tokens or required_output_tokens(
                    max(1, r["issues"]), max(1, r["parts"]),
                    thinking_reserve=thinking_reserve_for(r["first_effort"]))
                cut_vs_budget.setdefault(key, []).append((r["out_tok"], budget))
        modelled = verdict_content_tokens(max(1, r["issues"]), max(1, r["parts"]))
        by_model.setdefault(key, []).append((r["out_tok"], modelled, cut, repair))

    cut_rate: dict[str, float] = {}
    for model, quads in sorted(by_model.items()):
        # 평균에 넣는 건 **1차 호출이면서 잘리지 않은** 것뿐입니다.
        #   잘린 건  — 길이가 아니라 예산이 끝난 지점입니다.
        #   복구 건  — 짧게 쓰라고 지시받은 응답입니다.
        first = [(a, m, cut) for a, m, cut, rep in quads if not rep]
        clean = [(a, m) for a, m, cut in first if not cut]
        cuts = sum(1 for *_, cut in first if cut)
        repairs = sum(1 for *_, rep in quads if rep)
        cut_rate[model] = cuts / len(first) if first else 0.0
        if not clean:
            print(f"  {model:36} {len(first):>4} {cuts:>4} {repairs:>4} "
                  f"{'—':>9} {'—':>9} {'전부 잘림':>6}")
            continue
        actual = sum(a for a, _ in clean) / len(clean)
        modelled = sum(m for _, m in clean) / len(clean)
        ratio = actual / modelled if modelled else 0
        print(f"  {model:36} {len(first):>4} {cuts:>4} {repairs:>4} "
              f"{actual:>9,.0f} {modelled:>9,.0f} {ratio:>5.2f}x")

    merged = {k: v for k, v in spellings.items() if len(v) > 1}
    if merged:
        print("\n  접두사만 다른 표기를 합쳤습니다:")
        for key, raws in sorted(merged.items()):
            print(f"    {key} ← {', '.join(sorted(raws))}")

    print("\n  비율 1.00 이면 모델이 정확합니다. 1 보다 크면 모델이 낮게 잡는 것입니다.")
    print("  평균은 **1차 호출 중 잘리지 않은 건만** 씁니다 — 잘린 건은 예산이 끝난")
    print("  지점이고, 복구 건은 '짧게 쓰라'고 지시받은 응답이라 둘 다 길이의")
    print("  근거가 못 됩니다.")

    # 생존 편향. 잘림이 많을수록 남은 표본은 '마침 짧게 끝난 판정' 쪽으로
    # 쏠립니다. 이 경고가 없으면 그 평균으로 벤더 계수를 정하게 됩니다.
    biased = sorted(m for m, rate in cut_rate.items() if rate >= 0.2)
    if biased:
        print(f"\n  ⚠ 생존 편향: {', '.join(biased)} 는 1차 호출의 20% 이상이")
        print("  잘렸습니다. 잘린 건 길게 쓰려던 판정이므로, 남은 평균은 **실제보다")
        print("  짧습니다.** 얼마나 짧은지는 이 데이터로 알 수 없습니다 — 이 숫자를")
        print("  벤더 계수 근거로 쓰지 마십시오. 잘림을 먼저 고친 뒤 다시 재야 합니다.")

    if finishes:
        detail = ", ".join(f"{k!r}×{v}" for k, v in sorted(finishes.items()))
        print(f"\n  ⚠ 잘린 판정 호출: {detail}")
        # 원인을 가르는 숫자. 본문이 예산에 가까우면 진짜로 길이가 모자란 것이고
        # (예산 상향이 답), 한참 적으면 차액은 응답에 안 보이는 사고 토큰입니다
        # (예산만 올리면 사고가 그만큼 더 먹습니다).
        src = (f"DEBATE_JUDGE_MAX_TOKENS={settings.judge_max_tokens}"
               if settings.judge_max_tokens else "자동 계산(required_output_tokens)")
        for model, pairs in sorted(cut_vs_budget.items()):
            body = sum(o for o, _ in pairs) / len(pairs)
            budget = sum(b for _, b in pairs) / len(pairs)
            share = body / budget if budget else 0
            print(f"    {model}: 잘린 1차 {len(pairs)}건 — 본문 평균 {body:,.0f}tok "
                  f"/ 예산 {budget:,.0f}tok ({share:.0%})")
            if share < 0.5:
                print("      본문이 예산의 절반도 안 됩니다. 나머지는 응답에 안 보이는 "
                      "사고 토큰으로 보입니다 —")
                print("      예산 상향보다 사고 예산·수준을 낮추는 쪽이 원인에 닿습니다.")
            else:
                print("      본문이 예산 대부분을 썼습니다. 진짜로 길이가 모자란 것이라 "
                      "예산 상향이 답입니다.")
        print(f"  (예산은 현재 설정으로 다시 계산했습니다: {src}. 그 실행들 이후 "
              "설정을 바꿨다면 이 값은 틀립니다.)")

    # 여기까지는 **호출** 기준입니다. 사용자가 실제로 본 건 verdicts 입니다 —
    # 잘려도 복구에 성공하면 status='ok' 로 완결된 판정을 봅니다. 둘을 구분하지
    # 않으면 "판정 42% 가 잘려 보였다" 같은 틀린 결론이 나옵니다.
    # verdicts 에는 provider 가 없습니다. 그 토론의 **첫 판정 호출**에서 provider·
    # 모델·사고 수준을 가져옵니다. 예전 집계는 이 연결 없이 verdicts 를 통째로
    # 세서 fake 판정이 섞였습니다 — 실측: 판정 15건인데 실제 심판 1차 호출은 10건.
    seen = list(conn.execute("""
        WITH first_judge AS (
            SELECT debate_id, provider, model, reasoning_effort,
                   ROW_NUMBER() OVER (PARTITION BY debate_id ORDER BY rowid) AS seq
            FROM llm_calls WHERE purpose = 'judge'
        )
        SELECT v.status, v.finish_reason, v.parse_attempts,
               f.provider, f.model, f.reasoning_effort
        FROM verdicts v
        LEFT JOIN first_judge f ON f.debate_id = v.debate_id AND f.seq = 1
    """))
    if seen:
        fake_v = sum(1 for r in seen if r["provider"] == FAKE_PROVIDER)
        orphan = sum(1 for r in seen if r["provider"] is None)
        real = [r for r in seen if r["provider"] not in (None, FAKE_PROVIDER)]
        tally: dict[str, list[int]] = {}           # 키 → [정상, 복구, 실패]
        fails: dict[str, list[str]] = {}
        for r in real:
            key = _judge_key(r["model"], r["reasoning_effort"])
            row = tally.setdefault(key, [0, 0, 0])
            if r["status"] != "ok":
                row[2] += 1
                fails.setdefault(key, []).append(f"{r['status']}/{r['finish_reason']}")
            elif (r["parse_attempts"] or 1) > 1:
                row[1] += 1
            else:
                row[0] += 1

        print(f"\n사용자가 실제로 본 판정 {len(real)}건 (fake 제외)\n")
        print(f"  {'심판':40} {'정상':>4} {'복구':>4} {'실패':>4}")
        print("  " + "─" * 58)
        for key, (ok, rep, bad) in sorted(tally.items()):
            print(f"  {key:40} {ok:>4} {rep:>4} {bad:>4}"
                  + (f"   ({', '.join(fails[key])})" if bad else ""))
        print("\n  복구 = 1차가 잘려 '짧게' 다시 받은 판정. 점수는 유효하지만 길이는 줄여 받음.")
        if fake_v or orphan:
            parts = []
            if fake_v:
                parts.append(f"fake 심판 {fake_v}건")
            if orphan:
                parts.append(f"원장과 연결되지 않은 판정 {orphan}건(초기 debate_id 버그 시기)")
            print(f"  뺀 것: {', '.join(parts)}.")
    if len(by_model) > 1:
        print("\n  모델(벤더)마다 비율이 뚜렷이 다르면 프로바이더별 계수를 둘 근거가"
              " 됩니다.")
        print("  단, 주제가 다르면 길이도 달라집니다. 벤더 차이만 보려면 토론자를")
        print("  fake 로 고정하고 심판만 바꿔서 같은 주제로 여러 번 돌리십시오"
              " (README).")
    else:
        print(f"\n  비교하려면 다른 심판 모델로도 돌려야 합니다 (지금 {len(by_model)}종).")

    turns = list(conn.execute("""
        SELECT p.model, u.out_tok, LENGTH(u.content) AS chars
        FROM utterances u JOIN participants p
          ON p.debate_id = u.debate_id AND p.agent_id = u.agent_id
        WHERE u.status = 'ok' AND u.out_tok > 0 AND p.provider IS NOT ?
    """, (FAKE_PROVIDER,)))
    if turns:
        agg: dict[str, list[tuple[int, int]]] = {}
        for r in turns:
            agg.setdefault(_model_key(r["model"]), []).append((r["out_tok"], r["chars"]))

        print("\n발언 1건: 실측 (견적 가정 R1 250~450자 / R2+ 350~600자)\n")
        print(f"  {'모델':28} {'건수':>4} {'출력토큰':>8} {'글자':>7} {'토큰/자':>7}")
        print("  " + "─" * 66)
        tot_tok = tot_chars = 0
        for model, pairs in sorted(agg.items()):
            avg_out = sum(t for t, _ in pairs) / len(pairs)
            avg_chars = sum(c for _, c in pairs) / len(pairs)
            tot_tok += sum(t for t, _ in pairs)
            tot_chars += sum(c for _, c in pairs)
            per = avg_out / avg_chars if avg_chars else 0
            print(f"  {model:28} {len(pairs):>4} {avg_out:>8,.0f} "
                  f"{avg_chars:>7,.0f} {per:>7.2f}")
        if tot_chars:
            weighted = tot_tok / tot_chars
            print(f"\n  가중 평균 {weighted:.2f} 토큰/자 · 설정값 "
                  f"{settings.ko_tokens_per_char} (DEBATE_KO_TOKENS_PER_CHAR)")
            if weighted and abs(weighted - settings.ko_tokens_per_char) / weighted > 0.1:
                direction = "높게" if settings.ko_tokens_per_char > weighted else "낮게"
                print(f"  설정값이 {direction} 잡고 있습니다. 다만 **벤더가 한쪽으로"
                      " 쏠려 있으면** 그 벤더에")
                print("  맞추는 것일 뿐입니다 — 토크나이저가 다르면 값도 다릅니다.")
    conn.close()
    return 0


async def _cmd_estimate(args: argparse.Namespace) -> int:
    """LLM 을 한 번도 부르지 않고 비용 범위를 냅니다."""
    settings = Settings()
    fake = args.provider_override == FAKE_PROVIDER
    topic, rounds, specs = _collect_specs(args, fake)
    pricing = PricingTable.load(settings.pricing_path)

    judge = _resolve_judge(args, specs, settings)
    estimate = Estimator(pricing, settings.ko_tokens_per_char).estimate(
        participants=specs, rounds=rounds,
        judge_model=judge[1] if judge else None,
        moderator_model=args.moderator, topic=topic,
    )
    print(f"주제: {topic}")
    print(f"참가자 {len(specs)}명 | 라운드 {rounds}"
          + (f" | judge {judge[1]}" if judge else " | judge 없음"))
    print(estimate.format())
    print("llm_calls_made = 0")
    return 0


def _collect_specs(args, fake: bool) -> tuple[str, int, list[AgentSpec]]:
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
    return topic, rounds, specs


async def _cmd_run(args: argparse.Namespace) -> int:
    settings = Settings()
    fake = args.provider_override == FAKE_PROVIDER

    topic, rounds, specs = _collect_specs(args, fake)
    judge_spec = _resolve_judge(args, specs, settings)

    pricing = PricingTable.load(settings.pricing_path)
    debate_id = new_debate_id()
    meter = CostMeter(pricing, debate_id)
    global _current_budget
    _current_budget = settings.max_output_tokens
    # 심판도 풀에 들어가야 합니다. 예전에는 참가자 프로바이더만 조립해서,
    # 참가자와 다른 벤더를 심판으로 쓰면 **라운드를 다 돌린 뒤** 심판 단계에서
    # ConfigError 로 죽었습니다 (API 경로는 처음부터 넣고 있었습니다).
    real_for: set[str] = set()
    if judge_spec and judge_spec[0] != FAKE_PROVIDER:
        real_for.add(judge_spec[0])

    # fake 모드에서 심판만 진짜면 레지스트리가 필요합니다 — 통제 실험
    # (토론자 fake + 심판만 실제) 의 전제입니다.
    registry = provider_registry(settings) if (not fake or real_for) else None

    pool_specs = list(specs)
    if judge_spec and not fake:
        pool_specs.append(
            AgentSpec("__judge__", "__judge__", judge_spec[0], judge_spec[1], ""))

    pool = build_pool(
        pool_specs,
        registry,
        meter,
        settings,
        fake=_build_fake(specs, args.fake_latency, args.fail_agent,
                         args.fail_mode) if fake else None,
        dump=_make_dump(args.dump_raw),
        real_for=real_for,
    )

    concurrency = args.max_concurrency or settings.max_concurrency
    agents = [
        Agent(s, pool.get(s.provider), pricing,
              max_tokens=settings.max_output_tokens,
              timeout_s=settings.request_timeout_s)
        for s in specs
    ]
    cfg = DebateConfig(
        debate_id=debate_id,
        topic=topic,
        participants=tuple(specs),
        rounds=rounds,
        max_concurrency=concurrency,
        round_timeout_s=args.round_timeout or settings.round_timeout_s,
    )

    print(f"주제: {topic}")
    print(
        f"참가자 {len(specs)}명 | 라운드 {rounds} | 동시성 {concurrency} | "
        f"프로바이더 {'fake' if fake else ', '.join(sorted({s.provider for s in specs}))}"
    )
    estimate = Estimator(pricing, settings.ko_tokens_per_char).estimate(
        participants=specs, rounds=rounds,
        judge_model=judge_spec[1] if judge_spec else None,
        moderator_model=args.moderator, topic=topic,
    )
    print(estimate.format())

    warning = settings.timeout_warning()
    if warning:
        print(f"주의: {warning}")

    anonymizer = Anonymizer(specs)
    builder = ContextBuilder(anonymizer)
    moderator_model = args.moderator or specs[0].model
    moderator = Moderator(pool.get(specs[0].provider), moderator_model,
                          timeout_s=settings.request_timeout_s)

    if args.show_context:
        _preview_context(builder, specs, cfg)

    progress = _Progress({s.id: s.label for s in specs}, concurrency)
    verdict = None
    # 풀은 **판정까지 끝난 뒤에** 닫습니다. 심판도 같은 풀을 쓰므로 여기서
    # 먼저 닫으면 실제 HTTP 클라이언트가 "client has been closed" 로 죽습니다.
    # fake 프로바이더는 aclose() 가 no-op 이라 fake 검증으로는 안 잡힙니다.
    try:
        result = await DebateEngine(
            agents, builder, moderator, anonymizer, sink=progress
        ).run(cfg)

        anon_transcript = tuple(
            anonymizer.to_anon(u)
            for r in result.rounds for u in r.utterances if u.status == "ok"
        )
        if judge_spec and anon_transcript:
            judge = Judge(pool.get(judge_spec[0]), judge_spec[1],
                          max_tokens=settings.judge_max_tokens or None,
                          reasoning_effort=settings.judge_reasoning_effort or None,
                          timeout_s=settings.request_timeout_s)
            print(f"\n판정 중 — {judge_spec[1]} (쟁점 {len(result.issues)}개)", flush=True)
            started = time.perf_counter()
            verdict = await asyncio.wait_for(
                judge.evaluate(topic, result.issues, anon_transcript,
                               labels=[s.label for s in specs]),
                settings.judge_timeout_s)
            print(f"  판정 완료 ({time.perf_counter() - started:.1f}초)", flush=True)
    except asyncio.TimeoutError:
        print(f"\n판정 실패: {settings.judge_timeout_s:.0f}초 안에 응답하지 않았습니다 "
              "(DEBATE_JUDGE_TIMEOUT_S)", file=sys.stderr)
    finally:
        await pool.aclose()

    _print_result(result, meter)
    if verdict is not None:
        _print_verdict(verdict)

    note = None
    if judge_spec:
        note = family_note(judge_spec[1], [s.model for s in specs])
        # 경고가 아니라 사실 진술입니다. 모델 ID 비교만으로는 계열 내
        # 자기편애를 막지 못하므로, 막는 대신 관측 가능하게 둡니다.
        print(f"\n{note.line()}")

    rep = meter.report()
    print(f"견적 대비 실제: 예상 {_usd(estimate.low_usd)}~{_usd(estimate.high_usd)} "
          f"({estimate.tokens_low:,}~{estimate.tokens_high:,} tok) → "
          f"실제 {_usd(rep.total_usd)} ({rep.total_tokens:,} tok, {rep.calls}회)")

    if not args.no_store:
        store = SqliteStore(settings.db_path)
        try:
            store.save(result, meter, verdict, note)
            print(f"저장: {settings.db_path} (debate_id={result.debate_id})")
        finally:
            store.close()

    return 0 if result.status == "completed" else 1


def _cmd_serve(args) -> int:
    """API 서버를 띄웁니다. `DEBATE_HOST` / `DEBATE_PORT` 를 읽는 유일한 경로입니다.

    `python -m uvicorn debate.api:app` 로 직접 띄우면 uvicorn 은 `DEBATE_HOST` 를
    모릅니다. 그래서 설정만 추가하면 아무 효과가 없어, 그 값을 실제로 쓰는
    진입점을 같이 둡니다.

    워커는 1 로 고정합니다. 진행 중 토론은 `api._SESSIONS` (모듈 레벨 dict) 에만
    있어서, 워커가 2 개 이상이면 토론을 시작한 워커가 아닌 쪽에 `/stream` 이
    붙는 순간 404 가 납니다. 선택지로 두면 조용히 깨지는 구성이므로 막습니다.
    """
    try:
        import uvicorn
    except ImportError:
        print('uvicorn 이 없습니다. pip install -e ".[api]"', file=sys.stderr)
        return 2

    settings = Settings()
    host = args.host or settings.host
    port = args.port or settings.port
    loopback = host in {"127.0.0.1", "::1", "localhost"}

    print(f"bind {host}:{port} | workers=1 | config_ui={settings.config_ui}")
    if loopback:
        # `tailscale serve` 는 프록시가 127.0.0.1 로 붙으므로 루프백 바인드가
        # **올바른** 구성입니다. 예전 메시지는 이 경우에도 "다른 기기에서는 붙지
        # 않는다"고 단정해서, 제대로 맞춰 놓은 사람에게 틀렸다고 말했습니다.
        print("  루프백 바인드 — 이 주소로 직접 붙을 수 있는 건 이 컴퓨터뿐입니다.")
        print(f"  · `tailscale serve --bg {port}` 로 프록시 중이라면 이게 맞는 "
              "구성입니다 (프록시가 127.0.0.1 로 붙습니다).")
        print("  · 프록시 없이 폰에서 직접 붙으려면 DEBATE_HOST 를 tailscale IP 로 "
              "바꾸십시오.")
    else:
        # 요청만 보고는 바인드 주소를 알 수 없지만(api._require_local 참고),
        # 여기서는 우리가 바인드하는 쪽이라 확실히 압니다. 유일하게 이 사실을
        # 말할 수 있는 지점이므로 말합니다.
        print(f"  루프백이 아닌 주소에 바인드합니다. {host} 에 닿을 수 있는 "
              "모든 기기가 토론을 시작할 수 있습니다 — 당신의 쿼터로.")

    # 설정 UI 경고는 **바인드와 무관하게** 띄웁니다.
    #
    # 예전에는 루프백이 아닐 때만 띄웠는데, `tailscale serve` 를 쓰면 바인드는
    # 루프백인 채로 폰에서 닿습니다 — 경고가 가장 필요한 구성에서 침묵했습니다.
    # 바인드 주소만으로는 "밖에서 닿는가"를 알 수 없다는 게 이 프로젝트에서
    # 이미 확인된 사실이고(api._require_local), 그건 여기서도 똑같습니다.
    if settings.config_ui.lower() == "local":
        print("  경고: DEBATE_CONFIG_UI 가 'local' 입니다. 프록시나 터널을 앞에 "
              "두는 구성이라면 'off' 로 두십시오.", file=sys.stderr)
        print("        (루프백이 아닌 클라이언트는 _require_local 이 403 으로 "
              "막지만, 그건 두 번째 방어선입니다.)", file=sys.stderr)

    uvicorn.run("debate.api:app", host=host, port=port, workers=1)
    return 0


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
        help="'provider/model [| 입장 [| 페르소나]]'. 예: "
             "'openai/gpt-4o-mini | 반대 | 비용을 따지는 회의주의자'. 반복 지정.",
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
    r.add_argument("--judge", help="심판 모델 'provider/model'. 참가자와 같은 모델은 거부")
    r.add_argument("--allow-judge-overlap", action="store_true",
                   help="judge 가 참가자와 같은 모델이어도 허용 (점수 오염 감수)")
    r.add_argument("--no-store", action="store_true", help="sqlite 저장 생략")
    r.add_argument("--show-context", action="store_true",
                   help="마지막 라운드 컨텍스트를 실행 전에 출력")
    r.add_argument("--moderator", help="쟁점 추출·요약에 쓸 모델 (기본: 첫 참가자)")
    r.add_argument("--fail-agent", help="fake 모드 장애 주입: '모델' 또는 '모델@round2'")
    r.add_argument("--fail-mode", default="500", choices=["500", "429", "timeout", "fatal"])
    r.add_argument(
        "--dump-raw", metavar="DIR",
        help="프로바이더 원본 JSON 응답을 이 디렉터리에 저장 (헤더/키는 저장 안 함)",
    )
    r.set_defaults(fn=_cmd_run)

    e = sub.add_parser("estimate", help="LLM 호출 없이 비용 범위 계산")
    for flag in ("--topic", "--config", "--moderator", "--judge"):
        e.add_argument(flag)
    e.add_argument("--agent", action="append")
    e.add_argument("--rounds", type=int)
    e.add_argument("--allow-judge-overlap", action="store_true")
    e.add_argument("--provider-override", choices=[FAKE_PROVIDER])
    e.set_defaults(fn=_cmd_estimate)

    sv = sub.add_parser("serve", help="API 서버 (DEBATE_HOST/PORT 를 읽습니다)")
    sv.add_argument("--host", help="DEBATE_HOST 보다 우선")
    sv.add_argument("--port", type=int, help="DEBATE_PORT 보다 우선")
    sv.set_defaults(fn=_cmd_serve, sync=True)

    cal = sub.add_parser("calibrate",
                         help="기록된 실사용량을 견적 모델과 비교 (벤더별 비교 포함)")
    cal.set_defaults(fn=_cmd_calibrate, sync=True)

    args = parser.parse_args(argv)
    try:
        if getattr(args, "sync", False):
            return args.fn(args)
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
