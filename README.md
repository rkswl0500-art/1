# AI Debate Room

여러 AI가 하나의 주제로 다라운드 토론하고, 별도 Judge AI가 전체를 평가해 결론을 내리는
멀티 에이전트 시스템.

> **현재 상태: 슬라이스 1 (fake 모드) 완료. 라이브 스모크 대기 중.**
> 골격과 검증만 있습니다. Judge · 다라운드 · 쟁점 추출 · 저장 · API · UI 는 아직 없습니다.
> 실제 모델 호출 경로(`OpenAICompatProvider`)는 구현·테스트돼 있지만 **진짜 프로바이더로는
> 아직 한 번도 안 돌려봤습니다.** 아래 [라이브 스모크 3단계](#라이브-스모크-3단계)를
> 먼저 통과시킨 뒤 슬라이스 2 로 갑니다.

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
wall 2,101ms | sum(ok=2) 3,903ms | max 2,101ms | failed=0 | waves 1
```

- `sum(ok=N)` 은 **성공한 발언만** 합산합니다. 실패한 발언은 latency 가 0 이라
  합계에 섞으면 `wall > sum` 이 되어 병렬로 잘 돈 라운드가 직렬처럼 보입니다.
- `failed=0` 인 라운드에서만 다음이 성립합니다:
  `max ≤ wall < sum(ok)` → 병렬. `wall ≈ sum(ok)` → 직렬.
- `waves` = `ceil(참가자수 / 동시성)`. 동시성 3 에 참가자 5명이면 `waves 2` 가 되고
  `wall ≈ 2 × max` 입니다. **`wall ≈ max` 는 웨이브가 1일 때만 성립하는 기준입니다.**

`failed` 가 0 이 아니면 `wall` 을 병렬성 지표로 쓰지 마세요. 죽은 참가자가 재시도
백오프로 태운 시간(3회면 대략 3초)이 벽시계를 지배해서, 나머지가 아무리 잘 병렬로
돌아도 `wall > sum(ok)` 로 나옵니다:

```
wall 3,501ms | sum(ok=2) 3,302ms | max 1,801ms | failed=1 | waves 1
```

이 줄은 "직렬이었다"가 아니라 "한 명이 재시도에 3.5초를 태웠다"로 읽습니다.

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

## 라이브 스모크 3단계

fake 모드는 **내 가정이 일관되는지**만 검증합니다. 실제 서버가 그 가정대로
응답하는지는 검증하지 못합니다. 어긋난다면 슬라이스 2~3 을 쌓기 **전에** 알아야
하므로, 키를 채우는 즉시 아래를 순서대로 돌리세요. 슬롯 하나만 있으면 됩니다.

### 준비

`.env` 에 **슬롯 1개만** 채웁니다. 나머지 두 슬롯은 비워두세요 (NAME 과 BASE_URL 이
모두 비면 그냥 무시됩니다).

```bash
cp .env.example .env
```

```dotenv
DEBATE_PROVIDER_1_NAME=openai                    # 아무 이름이나. --agent 에서 이 이름을 씁니다
DEBATE_PROVIDER_1_BASE_URL=https://api.openai.com/v1
DEBATE_PROVIDER_1_API_KEY=sk-...                 # 실제 키
```

`BASE_URL` 은 **`/v1` 까지만** 씁니다. 끝 슬래시는 있어도 없어도 같습니다
(`/v1` 과 `/v1/` 모두 정상). `/chat/completions` 는 코드가 붙이므로 쓰지 마세요.

### 1단계 — 연결과 인증

```bash
.venv/bin/python -m debate.cli models --provider openai
```

**통과 조건**: 모델 ID 가 한 줄에 하나씩 출력되고 종료 코드 0.

```
gpt-4o-mini
gpt-4o
...
```

여기서 나온 ID 를 그대로 2단계에 씁니다. **눈으로 본 ID 만 쓰세요** — 기억이나
문서에서 가져온 이름은 대개 틀립니다. 이 단계는 BASE_URL · 키 · 네트워크를 한 번에
확인합니다.

### 2단계 — 실제 토론 1라운드

1단계에서 확인한 ID 로 2명을 세웁니다. 같은 프로바이더를 두 번 써도 됩니다.

```bash
.venv/bin/python -m debate.cli run \
    --topic "원격근무는 팀 생산성을 높이는가" \
    --agent openai/gpt-4o-mini \
    --agent openai/gpt-4o \
    --rounds 1
```

**통과 조건** — 네 가지가 모두 맞아야 합니다:

1. 참가자 A · B 의 **한국어** 발언이 출력됨
2. `failed=0`
3. `sum(ok=2) > wall` (병렬로 돌았다는 뜻)
4. 종료 코드 0

```
[R1] 참가자 A  (1,842ms, openai/gpt-4o-mini)
  원격근무의 생산성 효과를 측정하려면 먼저 ...

wall 2,118ms | sum(ok=2) 3,946ms | max 2,104ms | failed=0 | waves 1
[cost] calls=2  in=1,240  out=812  $0.0000  (가격 미상: ... — 0 으로 계상)
```

발언 안에 모델명·벤더명이 새어 나오는지도 같이 봐주세요. 있으면 그대로 알려주시면
슬라이스 2 의 스크럽 규칙에 반영합니다.

### 3단계 — 토큰 집계가 진짜인지

2단계 출력의 `[cost]` 줄만 다시 봅니다.

**통과 조건**: `in=` 과 `out=` 이 **둘 다 0 이 아님**.

일부 엔드포인트는 응답에 `usage` 를 담지 않습니다. 그러면 호출은 성공하고 종료 코드도
0 인데 토큰만 0 으로 집계됩니다 — **비용 추적이 조용히 죽는 유일한 경로**라서 따로
확인합니다. 0 이면 CLI 가 이렇게 경고합니다:

```
힌트: 호출은 성공했는데 토큰이 0 입니다. ...
```

이게 뜨면 슬라이스 3 의 견적·원장 설계를 바꿔야 하니 **반드시 알려주세요.**

`가격 미상` 경고는 정상입니다 — `config/pricing.yaml` 이 아직 비어 있어서 그렇습니다.
3단계에서 확인한 실제 모델 ID 로 채우면 사라집니다.

### 실패했을 때

`FatalError` 는 **재시도로 해결되지 않는 설정 문제**입니다. 참가자 단위로 격리되기
때문에 헤드라인이 `aborted_insufficient_participants` (종료 코드 1) 로 나오지만,
진짜 원인은 각 참가자 줄의 `── 실패:` 뒤에 있습니다. CLI 도 힌트를 같이 띄웁니다.

| 증상 | 원인 | 조치 |
|---|---|---|
| `FatalError: 401` | `API_KEY` 가 틀렸거나 비었음 | 키 재발급/재확인. 앞뒤 공백·따옴표 주의 |
| `FatalError: 403` | 키는 맞지만 해당 모델 권한 없음 | 콘솔에서 모델 접근 권한 확인 |
| `FatalError: 404` + `model ... does not exist` | 모델 ID 오타 | 1단계 출력에서 그대로 복사 |
| `FatalError: 404` + 그 외 | `BASE_URL` 에 `/v1` 누락 | `https://host/v1` 형태로 수정 |
| 요청이 `/v1/chat/completions/chat/completions` 로 감 | `BASE_URL` 에 `/chat/completions` 를 포함시킴 | `/v1` 까지만 남기기 |
| `ConfigError: 프로바이더 ... .env 에 없습니다` | `--agent` 의 이름 ≠ `DEBATE_PROVIDER_*_NAME` | 두 값을 일치시키기 |
| `RetryableError: 3회 시도 모두 실패` + `429` | 레이트리밋 | `--max-concurrency 1` 로 재시도 |
| `in=0 out=0` 인데 성공 | 엔드포인트가 `usage` 미제공 | 보고 필요 (위 3단계 참조) |

`ConfigError`(종료 코드 2) 는 **LLM 을 한 번도 부르기 전에** 납니다. 과금되지 않습니다.

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
| 1 | 생존 참가자가 2명 미만이라 토론 중단 |
| 2 | `ConfigError` — LLM 호출 0회, 과금 없음 |
| 3 | `ProviderError` — 토론 시작 전 단계(`models` 등)에서 터짐 |
| 4 | 미구현 경로 (예: 슬라이스 1 에서 다라운드 요청) |

**주의**: 토론 중의 인증 실패(401)는 3 이 아니라 **1** 로 나옵니다. 참가자 실패는
요구사항대로 개별 격리되기 때문에, 전원이 401 로 죽으면 결과적으로 "참가자 부족"이
됩니다. 진짜 원인은 각 참가자 줄과 CLI 힌트에 있습니다.

## 남은 슬라이스

| # | 범위 | 파일 |
|---|---|---|
| 2 | 다라운드 · 쟁점 3~5개 추출 · 컨텍스트 압축 · 익명화 스크럽 | `context.py`, `agent.py`(Moderator/Anonymizer) |
| 3 | Judge(단일 패스) · 사전 견적 · sqlite 저장 | `judge.py`, `cost.py`(Estimator), `storage.py` |
| 4 | FastAPI · SSE · 단일 HTML UI | `api.py`, `static/index.html` |

v1 범위에서 **뺀 것**: 조기 종료(합의 시 라운드 중단), Judge 의 참가자 수별 분기,
토론 재개. Judge 는 단일 패스로 고정하되 `prompt_tokens` 와 `finish_reason` 을
기록해서 실제로 잘리는 시점을 관측할 수 있게만 해둡니다.
