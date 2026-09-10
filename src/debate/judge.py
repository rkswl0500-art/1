"""Judge — 단일 패스 평가.

참가자 수에 따라 map-reduce 로 갈라지지 않습니다. v1 은 한 번의 호출로 고정하고,
대신 프롬프트 토큰 수와 finish_reason 을 기록해서 **실제로 잘리는 시점을 관측**
가능하게만 해둡니다. 5명에서 잘리는 게 확인되면 그때 바꿉니다.

편향 대응 두 가지:
  자기편애   참가자 풀에서 Judge 모델을 배제 (설정 검증 단계, ConfigError)
  위치편향   발언 제시 순서를 라운드마다 섞음 (LLM 은 처음/마지막을 우대)
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

from .models import AnonUtterance, ChatRequest, Issue, Message

RUBRIC_KEYS = ("근거", "논리", "반박", "명료성")
Margin = Literal["decisive", "narrow", "tie"]


@dataclass(frozen=True, slots=True)
class JudgeSpec:
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class IssueScore:
    issue_id: str
    scores: Mapping[str, int]      # 익명 라벨 -> 0..10
    reasoning: str


@dataclass(frozen=True, slots=True)
class Verdict:
    per_issue: tuple[IssueScore, ...]
    rubric: Mapping[str, Mapping[str, int]]
    winner: str | None
    margin: Margin
    conclusion: str
    #: 승자가 끝내 답하지 못한 가장 강한 반론. 이 칸이 없으면 Judge 는
    #: "양쪽 다 좋은 지적을 했습니다" 류의 무의미한 총평을 냅니다.
    dissent: str
    status: Literal["ok", "unparsed"] = "ok"
    raw: str = ""
    judge_model: str = ""
    prompt_tokens: int = 0
    #: 'length' 면 판정이 잘린 것입니다. 단일 패스를 언제 갈라야 할지의 신호.
    finish_reason: str | None = None
    parse_attempts: int = 1
    #: Judge 에게 실제로 보낸 프롬프트 전문.
    #:
    #: 저장해 두는 이유는 익명화를 **증명**하기 위해서입니다. 모델명이 안 샌다는
    #: 주장은 실제로 보낸 문자열을 grep 할 수 있어야 검증됩니다.
    prompt_text: str = ""

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def totals(self) -> dict[str, int]:
        return {label: sum(scores.values()) for label, scores in self.rubric.items()}


_PROMPT = """당신은 토론 심판입니다. 참가자가 아니며, 어느 발언이 어느 모델의
것인지 모릅니다. 참가자는 익명 라벨로만 식별됩니다.

«{topic}» 에 대한 토론 전문을 읽고 아래 쟁점별로 채점하십시오.

[쟁점]
{issues}

[채점 기준] 각 항목 0~10점
- 근거: 주장을 뒷받침하는 자료·사례의 質과 구체성
- 논리: 전제에서 결론까지의 연결이 타당한가
- 반박: 상대의 논점에 실제로 답했는가 (회피·화제전환은 감점)
- 명료성: 주장이 검증 가능한 형태로 진술되었는가

[지침]
- 발언 순서는 무작위입니다. 먼저 나온 쪽을 우대하지 마십시오.
- 길이가 아니라 내용으로 채점하십시오.
- "dissent" 에는 **승자가 끝내 답하지 못한 가장 강한 반론**을 쓰십시오.
  양비론이나 총평을 쓰지 마십시오.
- 최고점이 동률이면 winner 를 null, margin 을 "tie" 로 하십시오.

다른 말 없이 아래 JSON 만 출력하십시오.

{{"per_issue": [{{"issue_id": "i1", "scores": {{"참가자 A": 7}}, "reasoning": "..."}}],
  "rubric": {{"참가자 A": {{"근거": 7, "논리": 8, "반박": 6, "명료성": 7}}}},
  "winner": "참가자 A",
  "margin": "narrow",
  "conclusion": "...",
  "dissent": "..."}}"""


class Judge:
    def __init__(self, provider, model: str, *, max_tokens: int = 4096,
                 timeout_s: float = 180.0, seed: int = 0) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._rng = random.Random(seed)

    async def evaluate(
        self,
        topic: str,
        issues: Sequence[Issue],
        transcript: Sequence[AnonUtterance],
        labels: Sequence[str],
    ) -> Verdict:
        body = self._render_transcript(transcript)
        issue_text = (
            "\n".join(f"  {i.id}) {i.title}" for i in issues) if issues
            else "  (쟁점이 추출되지 않았습니다. 토론 전반을 평가하십시오.)"
        )
        system = _PROMPT.format(topic=topic, issues=issue_text)
        messages = [Message("system", system), Message("user", body)]
        sent = f"{system}\n\n{body}"

        raw, resp = "", None
        for attempt in (1, 2):
            resp = await self._provider.chat(ChatRequest(
                model=self._model, messages=tuple(messages),
                temperature=0.1, max_tokens=self._max_tokens,
                timeout_s=self._timeout_s, purpose="judge",
            ))
            raw = resp.text
            try:
                verdict = _parse(raw, labels, issues)
                return replace(
                    verdict, raw=raw, judge_model=resp.model,
                    prompt_tokens=resp.usage.prompt_tokens,
                    finish_reason=resp.finish_reason, parse_attempts=attempt,
                    prompt_text=sent,
                )
            except ValueError as e:
                if attempt == 2:
                    break
                # 복구 1회: 무엇이 틀렸는지 붙여 다시 물어봅니다.
                messages += [
                    Message("assistant", raw[:2000]),
                    Message("user", f"위 출력이 파싱되지 않았습니다: {e}\n"
                                    f"설명 없이 올바른 JSON 만 다시 출력하십시오."),
                ]

        # 두 번 실패해도 토론 기록은 살립니다. 판정만 unparsed 로 남깁니다.
        return Verdict(
            per_issue=(), rubric={}, winner=None, margin="tie",
            conclusion="", dissent="", status="unparsed", raw=raw,
            judge_model=resp.model if resp else self._model,
            prompt_tokens=resp.usage.prompt_tokens if resp else 0,
            finish_reason=resp.finish_reason if resp else None,
            parse_attempts=2, prompt_text=sent,
        )

    def _render_transcript(self, transcript: Sequence[AnonUtterance]) -> str:
        """라운드별로 제시 순서를 섞습니다.

        LLM 심판에는 위치 편향이 있습니다 — 먼저/나중에 제시된 쪽이 유리해집니다.
        라벨 매핑은 그대로 두고 순서만 흔들어 그 효과를 평균으로 보냅니다.
        """
        by_round: dict[int, list[AnonUtterance]] = {}
        for u in transcript:
            by_round.setdefault(u.round_no, []).append(u)

        blocks = []
        for round_no in sorted(by_round):
            shuffled = list(by_round[round_no])
            self._rng.shuffle(shuffled)
            said = "\n\n".join(f"{u.label}:\n{u.content}" for u in shuffled)
            blocks.append(f"=== 제{round_no}라운드 ===\n{said}")
        return "\n\n".join(blocks)


def _clamp(value) -> int:
    try:
        return max(0, min(10, int(value)))
    except (TypeError, ValueError):
        return 0


def _parse(raw: str, labels: Sequence[str], issues: Sequence[Issue]) -> Verdict:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"^```[a-z]*\n?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)
    except ValueError as e:
        raise ValueError(f"JSON 파싱 실패: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("최상위가 객체가 아닙니다")

    known = set(labels)
    per_issue = tuple(
        IssueScore(
            issue_id=str(item.get("issue_id") or f"i{n}"),
            scores={k: _clamp(v) for k, v in (item.get("scores") or {}).items()
                    if k in known},
            reasoning=str(item.get("reasoning") or ""),
        )
        for n, item in enumerate(data.get("per_issue") or [], start=1)
        if isinstance(item, dict)
    )
    rubric = {
        label: {k: _clamp(scores.get(k)) for k in RUBRIC_KEYS}
        for label, scores in (data.get("rubric") or {}).items()
        if label in known and isinstance(scores, dict)
    }
    if not rubric:
        raise ValueError(f"rubric 이 비었거나 라벨이 맞지 않습니다 (기대: {sorted(known)})")

    winner = data.get("winner")
    if winner is not None and winner not in known:
        raise ValueError(f"winner {winner!r} 가 참가자 라벨이 아닙니다")
    margin = data.get("margin")
    if margin not in ("decisive", "narrow", "tie"):
        margin = "tie" if winner is None else "narrow"

    return Verdict(
        per_issue=per_issue, rubric=rubric, winner=winner, margin=margin,
        conclusion=str(data.get("conclusion") or ""),
        dissent=str(data.get("dissent") or ""),
    )


# ── 계열 관측 (막지 않고 기록만) ─────────────────────────────────────────────


def vendor_family(model: str) -> str:
    """모델 ID 에서 벤더 계열을 추측합니다.

    **판정이 아니라 표시용입니다.** 이 값으로 아무것도 막지 않습니다 —
    계열 기준 자체가 애매하고(접두사? 벤더? OpenRouter 경유는?), 오탐이 나면
    정상 설정이 막힙니다. 두 번째 프로바이더 키가 생기면 같은 계열일 때와
    다를 때의 점수 분포를 실측으로 비교하는 게 추측으로 막는 것보다 낫습니다.
    """
    name = model.rsplit("/", 1)[-1].lower()
    head = re.split(r"[-_.:]", name)[0]
    return head or name


@dataclass(frozen=True, slots=True)
class FamilyNote:
    judge_family: str
    participant_families: tuple[str, ...]

    @property
    def shares_family(self) -> bool:
        return self.judge_family in self.participant_families

    def line(self) -> str:
        """사실 진술 한 줄. 경고가 아닙니다."""
        same = "참가자와 동일" if self.shares_family else "참가자와 다름"
        others = ", ".join(sorted(set(self.participant_families)))
        return f"judge: {self.judge_family} 계열 ({same}; 참가자 계열: {others})"


def family_note(judge_model: str, participant_models: Sequence[str]) -> FamilyNote:
    return FamilyNote(
        judge_family=vendor_family(judge_model),
        participant_families=tuple(vendor_family(m) for m in participant_models),
    )
