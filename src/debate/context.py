"""ContextBuilder — 요구사항 3(라운드 내 블라인드)과 6(컨텍스트 예산)의 심장.

컨텍스트 한 팩은 이렇게 구성됩니다:

    고정 헤더        주제 · 페르소나 · 익명 라벨 · 규칙 (라운드 불변)
    쟁점            R1 이후 확정. 이후 라운드는 이 범위 안에서만
    이전 요약        R1..R(n-2) 압축본
    직전 라운드 전문   R(n-1) 그대로, 익명화
    사회자 지시       이번 라운드 1회용 (슬라이스 4 에서 채워짐)

**진행 중인 라운드는 어디에서도 읽지 않습니다.** DebateState 에 그걸 담을
필드가 없어서 읽을 수가 없습니다 — 프롬프트로 부탁하는 게 아니라 자료구조로
막습니다. 이 파일에서 유일하게 신경 쓸 불변식이 그것입니다.
"""

from __future__ import annotations

from .agent import Anonymizer, build_header
from .models import AgentSpec, ContextPack, DebateState, Directive


class ContextBuilder:
    def __init__(self, anonymizer: Anonymizer) -> None:
        self._anon = anonymizer

    def build_for(self, spec: AgentSpec, state: DebateState) -> ContextPack:
        last = state.last_round
        last_anon = ()
        if last is not None:
            # 실패한 발언은 내용이 없으므로 컨텍스트에 넣지 않습니다.
            last_anon = tuple(
                self._anon.to_anon(u) for u in last.utterances if u.status == "ok"
            )

        return ContextPack(
            header=build_header(spec, state.topic),
            round_no=state.round_no,
            issues=state.issues,
            prior_digest=state.prior_digest,
            last_round=last_anon,
            directives=self._scrub_directives(state.directives),
        )

    def _scrub_directives(self, directives: tuple[Directive, ...]) -> tuple[Directive, ...]:
        """지시문도 스크럽을 통과시킵니다.

        사용자는 어느 참가자가 어느 모델인지 압니다. "Gemini가 말한 통계를
        캐물어라" 라고 쓰면 그 문자열이 컨텍스트에 들어가고, 슬라이스 3 에서는
        그대로 Judge 프롬프트까지 흘러갑니다. 사람이 넣은 텍스트라고 예외를
        두면 익명화에 구멍이 하나 생깁니다.
        """
        if not directives:
            return ()
        return tuple(
            Directive(
                text=self._anon.scrub(d.text).text,
                round_no=d.round_no,
                source=d.source,
            )
            for d in directives
        )
