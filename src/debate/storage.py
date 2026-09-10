"""토론 기록 저장. stdlib sqlite3 만 씁니다.

재개(resume)는 지원하지 않습니다 — 사후 기록만 필요하다는 결정에 따라 체크포인트
없이 완료 시점에 한 번 씁니다. 그래서 스키마도 진행 상태가 아니라 결과를 담는
모양입니다.

**anon_label 매핑은 여기에만 있습니다.** 어느 라벨이 어느 모델인지는 참가자
테이블에 기록되지만 프롬프트에는 절대 들어가지 않습니다. judge_prompt 컬럼에
실제로 보낸 프롬프트를 그대로 넣어두는 것도 그 사실을 grep 으로 증명하기
위해서입니다.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .cost import CostMeter
from .judge import FamilyNote, Verdict
from .models import DebateResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS debates (
    debate_id TEXT PRIMARY KEY, topic TEXT NOT NULL, status TEXT NOT NULL,
    rounds INTEGER NOT NULL, created_at TEXT NOT NULL,
    judge_model TEXT, judge_family TEXT, participant_families TEXT,
    judge_shares_family INTEGER, warnings TEXT
);
CREATE TABLE IF NOT EXISTS participants (
    debate_id TEXT NOT NULL, agent_id TEXT NOT NULL,
    anon_label TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
    persona TEXT, stance TEXT, dropped INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (debate_id, agent_id)
);
CREATE TABLE IF NOT EXISTS issues (
    debate_id TEXT NOT NULL, issue_id TEXT NOT NULL, title TEXT NOT NULL,
    PRIMARY KEY (debate_id, issue_id)
);
CREATE TABLE IF NOT EXISTS utterances (
    debate_id TEXT NOT NULL, round_no INTEGER NOT NULL, agent_id TEXT NOT NULL,
    content TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
    in_tok INTEGER, out_tok INTEGER, latency_ms INTEGER, cost_usd TEXT,
    finish_reason TEXT,
    PRIMARY KEY (debate_id, round_no, agent_id)
);
CREATE TABLE IF NOT EXISTS rounds (
    debate_id TEXT NOT NULL, round_no INTEGER NOT NULL,
    wall_ms INTEGER, sum_latency_ms INTEGER, waves INTEGER,
    ok_count INTEGER, failed_count INTEGER,
    PRIMARY KEY (debate_id, round_no)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY, debate_id TEXT NOT NULL, purpose TEXT NOT NULL,
    agent_id TEXT, provider TEXT, model TEXT, round_no INTEGER,
    in_tok INTEGER, out_tok INTEGER, cost_usd TEXT, priced INTEGER,
    latency_ms INTEGER, attempts INTEGER, finish_reason TEXT
);
CREATE TABLE IF NOT EXISTS verdicts (
    debate_id TEXT PRIMARY KEY, judge_model TEXT, status TEXT,
    winner TEXT, margin TEXT, conclusion TEXT, dissent TEXT,
    rubric TEXT, per_issue TEXT, totals TEXT,
    prompt_tokens INTEGER, finish_reason TEXT, parse_attempts INTEGER,
    judge_prompt TEXT, raw TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_debate ON llm_calls(debate_id, purpose);
"""


class Store(Protocol):
    def save(self, result: DebateResult, meter: CostMeter,
             verdict: Verdict | None, family: FamilyNote | None) -> None: ...


class SqliteStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, result: DebateResult, meter: CostMeter,
             verdict: Verdict | None = None, family: FamilyNote | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        dropped = set(result.dropped)
        with self._conn:                      # 하나의 트랜잭션 — 부분 저장 방지
            self._conn.execute(
                "INSERT OR REPLACE INTO debates VALUES (?,?,?,?,?,?,?,?,?,?)",
                (result.debate_id, result.topic, result.status, len(result.rounds), now,
                 verdict.judge_model if verdict else None,
                 family.judge_family if family else None,
                 json.dumps(sorted(set(family.participant_families)), ensure_ascii=False)
                 if family else None,
                 int(family.shares_family) if family else None,
                 json.dumps(list(result.warnings), ensure_ascii=False)),
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO participants VALUES (?,?,?,?,?,?,?,?)",
                [(result.debate_id, s.id, s.label, s.provider, s.model,
                  s.persona, s.stance, int(s.id in dropped)) for s in result.participants],
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO issues VALUES (?,?,?)",
                [(result.debate_id, i.id, i.title) for i in result.issues],
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO rounds VALUES (?,?,?,?,?,?,?)",
                [(result.debate_id, r.round_no, r.wall_ms, r.sum_latency_ms,
                  r.waves, r.ok_count, r.failed_count) for r in result.rounds],
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO utterances VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(result.debate_id, u.round_no, u.agent_id, u.content, u.status, u.error,
                  u.usage.prompt_tokens, u.usage.completion_tokens, u.latency_ms,
                  str(u.cost_usd), u.finish_reason)
                 for r in result.rounds for u in r.utterances],
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(c.call_id, c.debate_id, c.purpose, c.agent_id, c.provider, c.model,
                  c.round_no, c.usage.prompt_tokens, c.usage.completion_tokens,
                  str(c.cost_usd), int(c.priced), c.latency_ms, c.attempts,
                  c.finish_reason) for c in meter.records],
            )
            if verdict is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO verdicts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (result.debate_id, verdict.judge_model, verdict.status,
                     verdict.winner, verdict.margin, verdict.conclusion, verdict.dissent,
                     json.dumps(verdict.rubric, ensure_ascii=False),
                     json.dumps([{"issue_id": s.issue_id, "scores": dict(s.scores),
                                  "reasoning": s.reasoning} for s in verdict.per_issue],
                                ensure_ascii=False),
                     json.dumps(verdict.totals(), ensure_ascii=False),
                     verdict.prompt_tokens, verdict.finish_reason, verdict.parse_attempts,
                     verdict.prompt_text, verdict.raw),
                )

    def close(self) -> None:
        self._conn.close()
