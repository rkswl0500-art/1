# AI Debate Room

여러 AI가 하나의 주제로 다라운드 토론하고, 별도 Judge AI가 전체를 평가해 결론을 내리는
멀티 에이전트 시스템.

> **현재 상태: 슬라이스 1 (fake 모드) 완료.**
> 골격과 검증만 있습니다. Judge · 다라운드 · 쟁점 추출 · 저장 · API · UI 는 아직 없습니다.
> 실제 모델 호출 경로(`OpenAICompatProvider`)는 구현돼 있지만 키가 없어 미검증입니다.

## 레이어

```
Provider  →  Agent  →  DebateEngine  →  Judge  →  Storage  →  API
```

의존은 한 방향입니다: `models ← config ← cost ← provider ← agent ← engine ← cli`

핵심 규칙 세 가지:

- **인터페이스는 `ChatProvider` 하나.** 모델을 추가한다 = 설정에 한 줄 추가한다.
  토론 로직은 안 바뀝니다.
- **재시도와 계측은 데코레이터**(`RetryingProvider`, `MeteredProvider`)입니다.
  프로바이더 구현체마다 다시 짜지 않습니다.
- **라운드 내부는 병렬, 라운드 간은 순차.** 같은 라운드 참가자는 서로 못 봅니다 —
  프롬프트 훈계가 아니라 `ContextPack` 에 진행 중 라운드를 담을 필드가 아예 없어서
  구조적으로 불가능합니다.

## 설치

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

## fake 모드로 돌려보기 (키 불필요)

```bash
.venv/bin/python -m debate.cli models --provider fake

.venv/bin/python -m debate.cli run --provider-override fake \
    --topic "원격근무는 팀 생산성을 높이는가" \
    --agent fake-a --agent fake-b --rounds 1 --fake-latency 1800,2100
```

`FakeProvider` 는 테스트 픽스처가 아니라 1급 실행 모드입니다. 병렬성 · 재시도 ·
dropout · 컨텍스트 구성은 실제 모델보다 여기서 **더 정확하게** 검증됩니다 —
재현이 되니까요.

### 병렬성 읽는 법

```
wall 2,101ms | sum 3,903ms | max 2,101ms | waves 1
```

- `wall` ≈ `max`, `sum` 보다 훨씬 작음 → 라운드가 실제로 병렬
- `waves` = `ceil(참가자수 / 동시성)`. 동시성 3 에 참가자 5명이면 `waves 2` 가 되고
  `wall ≈ 2 × max` 입니다. **`wall ≈ max` 는 웨이브가 1일 때만 성립하는 기준입니다.**

## 실제 프로바이더 붙이기

```bash
cp .env.example .env      # 슬롯 3개(openai / groq / openrouter)에 키를 채우세요
```

게이트웨이 하나를 경유하지 않고 **참가자마다 다른 엔드포인트를 직접 호출**합니다.
`.env` 의 슬롯에 이름을 붙이고, 참가자가 그 이름을 참조합니다:

```yaml
# config/participants.yaml
participants:
  - id: p1
    provider: openai        # → DEBATE_PROVIDER_1_*
    model: gpt-4o-mini
  - id: p2
    provider: groq          # → DEBATE_PROVIDER_2_*
    model: llama-3.3-70b-versatile
```

```bash
.venv/bin/python -m debate.cli models --provider openai   # 모델 ID 확인
.venv/bin/python -m debate.cli run --config config/participants.yaml --rounds 1
```

슬롯은 1..9 까지 스캔합니다. `.env.example` 이 3개를 보여줄 뿐 코드가 3개로
제한하지 않습니다. `API_KEY` 가 비면 `Authorization` 헤더를 아예 안 붙이므로
인증 없는 로컬 엔드포인트(vLLM · Ollama · LM Studio)도 그대로 붙습니다.

## 키 취급

- `.env` 는 `.gitignore` 에 있습니다. 커밋되지 않습니다.
- 키는 `SecretStr` 이고 `ProviderSlot.__repr__` 이 재정의돼 있어 로그·트레이스백에
  찍히지 않습니다 (`test_api_key_never_appears_in_repr`).
- 슬라이스 4의 API 응답에도 키·`base_url` 이 나가지 않습니다.

## 비용

- 단가는 `config/pricing.yaml`. **가격 미상 모델은 0 으로 계상하고 경고만 냅니다 —
  절대 죽지 않습니다.**
- 가격표에 `0` 이라고 적힌 것(= 무료임을 앎)과 표에 없는 것(= 모름)은 구분됩니다.
  후자만 `가격 미상` 으로 경고합니다.
- 한국어 토큰 추정 계수 `DEBATE_KO_TOKENS_PER_CHAR` 는 **견적 전용**입니다.
  원장에 남는 실사용량은 언제나 API 응답의 `usage` 라서 계수가 틀려도 기록은 정확합니다.
  슬라이스 3 에서 estimate vs actual 로 보정합니다.

## 테스트

```bash
.venv/bin/python -m pytest tests/ -q
```

## 종료 코드

| 코드 | 의미 |
|---|---|
| 0 | 정상 종료 |
| 1 | 참가자 부족으로 토론 중단 |
| 2 | `ConfigError` — LLM 호출 0회 |
| 3 | `ProviderError` |
| 4 | 미구현 경로 (예: 슬라이스 1 에서 다라운드 요청) |

## 남은 슬라이스

| # | 범위 | 파일 |
|---|---|---|
| 2 | 다라운드 · 쟁점 3~5개 추출 · 컨텍스트 압축 · 익명화 스크럽 | `context.py`, `agent.py`(Moderator/Anonymizer) |
| 3 | Judge(단일 패스) · 사전 견적 · sqlite 저장 | `judge.py`, `cost.py`(Estimator), `storage.py` |
| 4 | FastAPI · SSE · 단일 HTML UI | `api.py`, `static/index.html` |

v1 범위에서 **뺀 것**: 조기 종료(합의 시 라운드 중단), Judge 의 참가자 수별 분기,
토론 재개. Judge 는 단일 패스로 고정하되 `prompt_tokens` 와 `finish_reason` 을
기록해서 실제로 잘리는 시점을 관측할 수 있게만 해둡니다.
