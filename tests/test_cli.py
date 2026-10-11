"""CLI 진단 출력 테스트.

라이브 스모크에서 사용자가 헛다리를 짚지 않으려면 두 상황이 눈에 띄어야 합니다:
설정 오류(FatalError)와 usage 미제공.
"""

from __future__ import annotations

import debate.cli as cli
from debate.provider import FakeBehavior, FakeProvider


def _run(monkeypatch, fake: FakeProvider, argv: list[str]) -> int:
    monkeypatch.setattr(cli, "_build_fake", lambda *a, **k: fake)
    return cli.main(argv)


BASE = ["run", "--provider-override", "fake", "--topic", "T",
        "--agent", "m1", "--agent", "m2"]


def test_fatal_error_prints_config_hint(monkeypatch, capsys):
    """401 은 참가자 단위로 격리되는 바람에 헤드라인이 '참가자 부족'으로 보입니다.
    실제 원인이 키/모델 ID 라는 걸 짚어주지 않으면 엉뚱한 곳을 뒤지게 됩니다."""
    fake = FakeProvider({
        "m1": FakeBehavior(latency_ms=1, fail_always=True, fail_mode="fatal"),
        "m2": FakeBehavior(latency_ms=1, fail_always=True, fail_mode="fatal"),
    })
    code = _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert code == 1
    assert "aborted_insufficient_participants" in out
    assert "FatalError 는 재시도로 해결되지 않는" in out
    assert "API_KEY" in out and "BASE_URL" in out


def test_transient_failure_does_not_print_config_hint(monkeypatch, capsys):
    """500 은 설정 문제가 아니므로 힌트를 띄우면 오히려 오도합니다."""
    fake = FakeProvider({
        "m1": FakeBehavior(latency_ms=1),
        "m2": FakeBehavior(latency_ms=1, fail_always=True, fail_mode="500"),
    })
    _run(monkeypatch, fake, BASE)
    assert "재시도로 해결되지 않는" not in capsys.readouterr().out


def test_zero_usage_is_flagged(monkeypatch, capsys):
    """usage 를 안 돌려주는 엔드포인트는 성공(exit 0)하면서 집계만 0 이 됩니다.
    비용 추적이 조용히 죽는 유일한 경로라 반드시 경고해야 합니다."""
    class NoUsage(FakeProvider):
        async def chat(self, req):
            from dataclasses import replace
            from debate.models import Usage
            return replace(await super().chat(req), usage=Usage(0, 0))

    code = _run(monkeypatch, NoUsage({"m1": FakeBehavior(latency_ms=1),
                                      "m2": FakeBehavior(latency_ms=1)}), BASE)
    out = capsys.readouterr().out

    assert code == 0                      # 실패가 아님 — 그래서 더 놓치기 쉬움
    assert "토큰이 0 입니다" in out


def test_healthy_run_prints_no_hints(monkeypatch, capsys):
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=1), "m2": FakeBehavior(latency_ms=1)})
    code = _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert code == 0
    assert "힌트:" not in out
    assert "failed=0" in out


def test_truncated_utterance_is_flagged_loudly(monkeypatch, capsys):
    """토론에서 잘린 발언은 곧 잘못된 판정입니다. 기록만 하고 안 보여주면
    사용자는 Judge 결과가 왜 이상한지 영원히 모릅니다."""
    class Truncating(FakeProvider):
        async def chat(self, req):
            from dataclasses import replace
            return replace(await super().chat(req), finish_reason="length")

    code = _run(monkeypatch, Truncating({"m1": FakeBehavior(latency_ms=1),
                                         "m2": FakeBehavior(latency_ms=1)}), BASE)
    out = capsys.readouterr().out

    assert code == 0                                  # 실패가 아니라서 더 위험
    assert "⚠ 잘림(finish_reason=length)" in out
    assert "발언이 잘렸습니다" in out
    assert "DEBATE_MAX_OUTPUT_TOKENS" in out


def test_normal_finish_reason_prints_no_truncation_warning(monkeypatch, capsys):
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=1), "m2": FakeBehavior(latency_ms=1)})
    _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert "잘림" not in out
    assert "finish_reason='stop'" in out              # 항상 보이게


def test_progress_shows_fast_agent_while_slow_one_runs(monkeypatch, capsys):
    """지연 편차가 30배(관측: 54.3s vs 1.8s)라 라운드는 가장 느린 참가자에
    묶입니다. 빠른 쪽 완료가 즉시 보여야 멈춘 게 아니라 기다리는 중임을 압니다."""
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=200),
                         "m2": FakeBehavior(latency_ms=5)})
    _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert "[R1] 시작" in out
    assert "참가자 B 완료" in out and "[1/2]" in out    # 빠른 쪽이 먼저
    assert out.index("참가자 B 완료") < out.index("참가자 A 완료")


def test_timeout_warning_silent_on_defaults(monkeypatch, capsys):
    """기본값에서 뜨는 경고는 경고가 아니라 노이즈입니다."""
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=1), "m2": FakeBehavior(latency_ms=1)})
    _run(monkeypatch, fake, BASE)
    assert "라운드 타임아웃" not in capsys.readouterr().out


def test_timeout_warning_fires_on_inconsistent_override(monkeypatch, capsys):
    monkeypatch.setenv("DEBATE_ROUND_TIMEOUT_S", "60")
    fake = FakeProvider({"m1": FakeBehavior(latency_ms=1), "m2": FakeBehavior(latency_ms=1)})
    _run(monkeypatch, fake, BASE)
    out = capsys.readouterr().out

    assert "라운드 타임아웃 60s < 재시도 예산 360s" in out


def test_agent_flag_accepts_stance_and_persona():
    from debate.cli import _parse_agent_flag

    spec = _parse_agent_flag("openai/gpt-4o-mini | 반대 | 비용을 따지는 회의주의자",
                             0, fake=False)

    assert (spec.provider, spec.model) == ("openai", "gpt-4o-mini")
    assert spec.stance == "반대"
    assert spec.persona == "비용을 따지는 회의주의자"
    assert spec.label == "참가자 A"


def test_agent_flag_without_stance_still_works():
    from debate.cli import DEFAULT_PERSONA, _parse_agent_flag

    spec = _parse_agent_flag("openai/gpt-4o-mini", 1, fake=False)

    assert spec.stance is None
    assert spec.persona == DEFAULT_PERSONA


# ── calibrate: 측정 도구가 거짓 수치를 내지 않는가 ──────────────────────────

def _history(db_path):
    """실제 히스토리를 흉내 낸 DB — fake 섞임, 접두사 두 표기, 잘린 판정."""
    import sqlite3

    from debate.storage import SqliteStore

    SqliteStore(db_path).close()                      # 스키마만
    c = sqlite3.connect(db_path)

    def debate(did, prov="gemini"):
        c.execute("INSERT INTO debates (debate_id, topic, status, rounds, created_at)"
                  " VALUES (?,?,?,?,?)", (did, "T", "completed", 2, "2026-10-01"))
        for i in range(3):
            c.execute("INSERT INTO issues VALUES (?,?,?)", (did, f"i{i}", "쟁점"))
        for i in range(2):
            c.execute("INSERT INTO participants VALUES (?,?,?,?,?,?,?,?)",
                      (did, f"p{i}", "참가자 A", prov, "models/flash-lite"
                       if i == 0 else "flash-lite", "", "", 0))

    def call(did, purpose, provider, model, out, finish="stop"):
        c.execute("INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (f"{did}{purpose}{model}{out}{finish}", did, purpose, "p0", provider,
                   model, 1, 1000, out, "0", 1, 500, 1, finish))

    def turn(did, agent, out, chars):
        c.execute("INSERT INTO utterances VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (did, 1, agent, "가" * chars, "ok", None, 1000, out, 500, "0", "stop"))

    # 판정은 토론당 1건이 정상입니다. 한 토론에 2건이면 뒤엣것이 복구 호출이라,
    # 토론을 나눠야 '1차 3건' 이라는 실제 형태가 됩니다.
    debate("d_r1")
    call("d_r1", "judge", "gemini", "models/flash-preview", 1150)
    turn("d_r1", "p0", 260, 420)

    debate("d_r2")
    call("d_r2", "judge", "gemini", "flash-preview", 1190)        # 같은 모델, 다른 표기

    debate("d_r3")                                                 # 잘리고 복구 못 함
    call("d_r3", "judge", "gemini", "flash-preview", 408, finish="length")

    debate("d_r4")                                                 # 잘렸다가 복구 성공
    call("d_r4", "judge", "gemini", "flash-preview", 410, finish="length")
    call("d_r4", "judge", "gemini", "flash-preview", 520)          # ← 복구(짧게 지시받음)

    def verdict(did, status="ok", finish="stop", attempts=1):
        c.execute("INSERT INTO verdicts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (did, "flash-preview", status, "참가자 A", "narrow", "", "",
                   "{}", "[]", "{}", 1900, finish, attempts, "", ""))

    verdict("d_r1")                                   # 그냥 정상
    verdict("d_r2")
    verdict("d_r3", status="unparsed", finish="length", attempts=2)   # 사용자가 본 실패
    verdict("d_r4", attempts=2)                       # 복구 성공 — 사용자는 완결본을 봄

    debate("d_fake", prov="fake")                                  # fake 실행
    call("d_fake", "judge", "fake", "models/flash-preview", 156)
    turn("d_fake", "p0", 130, 161)
    c.commit()
    c.close()


def _calibrate(tmp_path, monkeypatch, capsys) -> str:
    db = tmp_path / "debates.db"
    _history(db)
    monkeypatch.setenv("DEBATE_DB_PATH", str(db))
    assert cli.main(["calibrate"]) == 0
    return capsys.readouterr().out


def test_calibrate_excludes_fake_runs_and_says_so(tmp_path, monkeypatch, capsys):
    """fake 모드는 모델명을 고른 그대로 남깁니다 — 그 설정을 리허설했다는
    기록이라 지우지 않습니다. 대신 provider 로 걸러야 합니다.

    예전 쿼리는 모델명으로만 묶어서, fake 출력(156tok)이 실제 심판 모델의
    실측치로 보고됐습니다. 측정 도구가 거짓 합격을 내는 건 측정 안 한 것보다
    나쁩니다.
    """
    out = _calibrate(tmp_path, monkeypatch, capsys)

    assert "fake 실행 1건을 뺐습니다" in out        # 조용히 거르지 않습니다
    assert "156" not in out                        # fake 판정 길이가 안 섞입니다
    assert "161" not in out                        # fake 발언 글자수도


def test_calibrate_merges_prefix_spellings(tmp_path, monkeypatch, capsys):
    """`models/x` 와 `x` 가 따로 집계되면 건수가 반으로 보이고 평균도 갈립니다."""
    out = _calibrate(tmp_path, monkeypatch, capsys)

    assert "접두사만 다른 표기를 합쳤습니다" in out
    assert "flash-preview ← flash-preview, models/flash-preview" in out


def test_calibrate_separates_truncated_verdicts(tmp_path, monkeypatch, capsys):
    """잘린 판정을 평균에 섞으면 '간결한 모델'로 오독됩니다.

    1차 4건 중 1,150 / 1,190 은 정상, 408 과 410 은 length 입니다. 넷을 섞으면
    790 — 실제보다 32% 낮습니다. 그 숫자로 벤더 성향을 논하면 버그를 성향으로
    착각합니다.
    """
    out = _calibrate(tmp_path, monkeypatch, capsys)

    assert "1,170" in out                          # (1150+1190)/2
    assert "790" not in out                        # 넷을 섞은 값
    assert "'length'×2" in out


def test_calibrate_excludes_repair_calls_from_the_average(tmp_path, monkeypatch, capsys):
    """복구 호출은 '짧게 쓰라'고 지시받은 응답이라 길이의 근거가 못 됩니다.

    d_r4 의 520 은 복구 건입니다. 평균에 들어가면 (1150+1190+520)/3 = 953 이
    되어, 잘린 건을 뺀 효과가 복구 건 때문에 도로 사라집니다.
    """
    out = _calibrate(tmp_path, monkeypatch, capsys)

    assert "953" not in out
    assert "1,170" in out


def test_calibrate_warns_about_survivorship_bias(tmp_path, monkeypatch, capsys):
    """잘림이 많으면 남은 평균은 '마침 짧게 끝난 판정' 쪽으로 쏠립니다.

    경고가 없으면 그 평균으로 벤더 계수를 정하게 됩니다 — 측정 도구가 틀린
    결론을 거들어서는 안 됩니다.
    """
    out = _calibrate(tmp_path, monkeypatch, capsys)       # 1차 4건 중 2건 잘림 = 50%

    assert "생존 편향" in out
    assert "벤더 계수 근거로 쓰지 마십시오" in out


def test_calibrate_separates_what_users_actually_saw(tmp_path, monkeypatch, capsys):
    """`length` 호출이 있다고 사용자가 잘린 판정을 본 건 아닙니다.

    복구에 성공하면 verdicts 는 status='ok' 이고 사용자는 완결된 판정을 봅니다.
    호출 기준과 판정 기준을 구분하지 않으면 "판정 42% 가 잘려 보였다" 같은
    틀린 결론이 나옵니다.
    """
    out = _calibrate(tmp_path, monkeypatch, capsys)

    assert "사용자가 실제로 본 판정" in out


def test_model_key_normalizes_vendor_prefixes():
    assert cli._model_key("models/gemini-3.5-flash-lite") == "gemini-3.5-flash-lite"
    assert cli._model_key("openai/gpt-oss-120b") == "gpt-oss-120b"
    assert cli._model_key("gemini-3.5-flash-lite") == "gemini-3.5-flash-lite"
