# AI Debate Room

여러 AI가 하나의 주제로 다라운드 토론하고, 별도 Judge AI가 전체를 평가해 결론을 내리는
멀티 에이전트 시스템.

> **현재 상태: 슬라이스 3 완료.** 라이브 스모크는 슬라이스 1 시점에 통과했습니다.
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

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

`Activate.ps1` 이 실행 정책 때문에 막히면 아래를 한 번 치고 다시 하세요.
현재 창에만 적용되고 시스템 설정은 안 건드립니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**이 아래의 모든 명령은 venv 가 활성화된 상태를 가정합니다.** 활성화 없이 쓰려면
`python` 자리에 macOS/Linux 는 `.venv/bin/python`, Windows 는
`.venv\Scripts\python` 을 넣으세요.

## fake 모드로 돌려보기 (키 불필요)

```bash
python -m debate.cli models --provider fake

python -m debate.cli run --provider-override fake --topic "원격근무는 팀 생산성을 높이는가" --agent fake-a --agent fake-b --rounds 1 --fake-latency 1800,2100
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
cp .env.example .env      # PowerShell: copy .env.example .env
                          # 슬롯 3개(openai / groq / openrouter)에 키를 채우세요
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
python -m debate.cli models --provider openai   # 모델 ID 확인
python -m debate.cli run --config config/participants.yaml --rounds 1
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
cp .env.example .env          # PowerShell: copy .env.example .env
```

```dotenv
DEBATE_PROVIDER_1_NAME=openai                    # 아무 이름이나. --agent 에서 이 이름을 씁니다
DEBATE_PROVIDER_1_BASE_URL=https://api.openai.com/v1
DEBATE_PROVIDER_1_API_KEY=sk-...                 # 실제 키
```

`BASE_URL` 은 **`/v1` 까지만** 씁니다. 끝 슬래시는 있어도 없어도 같습니다
(`/v1` 과 `/v1/` 모두 정상). `/chat/completions` 는 코드가 붙이므로 쓰지 마세요.

**`.env` 는 명령을 실행하는 디렉터리에서 찾습니다.** 프로젝트 루트에서 돌리세요.
못 찾으면 에러가 어느 절대 경로를 봤는지 알려줍니다.

BOM 은 신경 쓰지 않아도 됩니다. PowerShell 의 `Out-File -Encoding utf8` 은 기본으로
BOM 을 붙이는데, `.env` 를 utf-8-sig 로 읽으므로 있든 없든 동작합니다.

### 1단계 — 연결과 인증

```bash
python -m debate.cli models --provider openai
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

```
python -m debate.cli run --topic "원격근무는 팀 생산성을 높이는가" --agent openai/gpt-4o-mini --agent openai/gpt-4o --rounds 1
```

한 줄이 길어서 쪼개고 싶다면 이어쓰기 문자가 셸마다 다릅니다 — bash 는 `\`,
PowerShell 은 백틱입니다. 헷갈리면 그냥 한 줄로 쓰세요.

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

### 발언이 잘리거나 이상할 때 — 원본 응답 받기

응답 파싱이 의심되면 원본 JSON 을 파일로 남겨서 보세요. 잘림이 모델 쪽인지
파싱 쪽인지는 원본을 봐야만 갈립니다.

```
python -m debate.cli run --topic "..." --agent gemini/<ID> --agent gemini/<ID> --rounds 1 --dump-raw ./raw
```

`./raw/01-*.json` 에 보낸 메시지와 받은 JSON 본문이 그대로 들어갑니다.
**요청 헤더는 저장하지 않습니다** — `Authorization` 이 파일에 남지 않게 하려는
것이라, 덤프를 그대로 공유해도 키는 새지 않습니다.

각 발언 아래에 `└ N자 / finish_reason=...` 이 항상 찍힙니다. 읽는 법:

| finish_reason | 뜻 |
|---|---|
| `'stop'` | 모델이 스스로 끝냈습니다. 짧다면 그건 모델의 선택입니다 |
| `'length'` | **예산에서 잘렸습니다.** `DEBATE_MAX_OUTPUT_TOKENS` 를 올리세요 |
| `None` | 엔드포인트가 이 필드를 안 줍니다. 잘림 여부를 알 수 없습니다 |

본문이 비어 있으면 성공으로 통과시키지 않고 실패로 처리합니다. 빈 발언을
그냥 두면 Judge 가 그걸 "논거 없음"으로 채점해서 결과가 조용히 망가집니다.

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
| `프로바이더를 찾을 수 없습니다` + `사용 가능: <없음>` + `이 경로에 파일이 없습니다` | 프로젝트 루트가 아닌 곳에서 실행 | 에러가 찍은 경로에 `.env` 를 두거나 루트에서 실행 |
| `프로바이더를 찾을 수 없습니다` + `사용 가능: gemini` | `--agent` 의 이름 ≠ `DEBATE_PROVIDER_*_NAME` | 두 값을 일치시키기 (에러가 인식된 이름을 보여줍니다) |
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

## 판정과 비용

```
python -m debate.cli estimate --config config/participants.yaml --judge gemini/<ID>
python -m debate.cli run --config config/participants.yaml --judge gemini/<ID>
```

`estimate` 는 **LLM 을 한 번도 부르지 않습니다** (`llm_calls_made = 0`). 비용을
보고 시작 여부를 정하라고 생성과 실행이 나뉘어 있습니다. 견적은 **범위**로
나옵니다 — 출력 길이를 모르는데 단일 숫자를 내면 그건 거짓말입니다.

Judge 는 **참가자와 같은 모델이면 실행 전에 거부합니다**(`ConfigError`, 과금 0).
모델이 부족하면 `--allow-judge-overlap` 으로 명시적으로 허용할 수 있습니다.

편향 대응 두 가지가 들어가 있습니다:

- **자기편애** — Judge 모델을 참가자 풀에서 배제. 발언은 익명 라벨로만 전달.
- **위치 편향** — LLM 심판은 먼저/나중에 제시된 쪽을 우대하므로, 라운드마다
  제시 순서를 섞습니다.

Judge 는 참가자 수와 무관하게 **단일 패스**입니다. 대신 `prompt_tokens` 와
`finish_reason` 을 기록해서 실제로 잘리는 시점을 관측만 합니다 — 5명에서 잘리는
게 확인되면 그때 map-reduce 로 가릅니다.

## 저장된 기록 보기

`data/debates.db` (sqlite). `sqlite3` CLI 가 없으면 파이썬으로 보면 됩니다:

```python
import sqlite3
c = sqlite3.connect("data/debates.db"); c.row_factory = sqlite3.Row

# 익명화 검증 — Judge 에게 실제로 보낸 프롬프트를 그대로 검사
prompt = c.execute("select judge_prompt from verdicts").fetchone()[0]
models = [r[0] for r in c.execute("select distinct model from participants")]
print([m for m in models if m in prompt])      # [] 여야 정상

# 라벨 -> 모델 매핑 (DB 에만 있고 프롬프트에는 없음)
print(dict(c.execute("select anon_label, model from participants")))

# 용도별 · 라운드별 비용
for r in c.execute("select purpose, count(*), sum(in_tok), sum(out_tok)"
                   " from llm_calls group by purpose"):
    print(tuple(r))
```

재개(resume)는 지원하지 않습니다. 사후 기록만 필요하다는 결정에 따라 체크포인트
없이 완료 시점에 한 번 씁니다.

## 사람의 개입 (슬라이스 4 예정, 자리만 뚫려 있음)

토론 중간에 사람이 지시를 넣는 기능은 아직 **동작하지 않습니다.** 다만 나중에
끼울 자리는 이미 있고, 지금은 항상 비어 있습니다:

| 자리 | 위치 | 현재 |
|---|---|---|
| `ContextPack.directives` | 쟁점 뒤, 차례 지시 앞에 렌더 | 항상 빈 튜플 |
| `InterventionGate` | 매 라운드 시작 전 엔진이 호출 | `NoIntervention` (즉시 빈 튜플) |
| 지시문 스크럽 | `ContextBuilder._scrub_directives` | 참가자 발언과 동일 경로 |
| 지시 흔적 | `DebateState.directive_trace` → 요약 프롬프트 | 비어 있으면 규칙 자체가 빠짐 |

확정된 규칙:

- **1회용.** 지정된 라운드에만 들어가고, 이후에는 요약본에 "사회자가 X를
  지시함" 한 줄로만 남습니다. 누적하면 라운드가 갈수록 지시문이 헤더처럼 쌓입니다.
- **전체 브로드캐스트만.** 참가자별 개별 지시는 컨텍스트를 참가자마다 갈라놓아
  Judge 가 "왜 이 참가자만 이 얘길 하지"를 판단할 근거를 잃습니다.
- **지시문도 스크럽을 통과합니다.** 사용자는 어느 참가자가 어느 모델인지 알기
  때문에, 지시문이 모델명 누설의 가장 쉬운 경로입니다.

## 테스트

```bash
python -m pytest tests/ -q
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
| ~~2~~ | ~~다라운드 · 쟁점 추출 · 컨텍스트 압축 · 익명화 스크럽~~ (완료) | `context.py`, `agent.py` |
| ~~3~~ | ~~Judge(단일 패스) · 사전 견적 · sqlite 저장~~ (완료) | `judge.py`, `cost.py`, `storage.py` |
| 4 | FastAPI · SSE · 단일 HTML UI | `api.py`, `static/index.html` |

v1 범위에서 **뺀 것**: 조기 종료(합의 시 라운드 중단), Judge 의 참가자 수별 분기,
토론 재개. Judge 는 단일 패스로 고정하되 `prompt_tokens` 와 `finish_reason` 을
기록해서 실제로 잘리는 시점을 관측할 수 있게만 해둡니다.

### 알려진 한계 — Judge 격리는 계열 내 자기편애를 막지 못합니다

`allow_judge_model_overlap=False` 검사는 **모델 ID 만 비교**합니다. 그래서

    참가자: gemini-2.x-flash   +   Judge: gemini-2.x-pro

같은 조합은 **검사를 통과합니다.** 그런데 이 규칙이 애초에 막으려던 것은
자기편애이고, 그건 같은 벤더·같은 계열 모델 사이에서도 상당 부분 남습니다.
즉 **검사는 통과하는데 목적은 달성 못 하는 상태**입니다.

프로바이더 키가 하나뿐이면 참가자도 Judge 도 전부 한 계열이 되므로 실제로 자주
걸립니다. 슬라이스 3 에서 두 갈래 중 하나를 골라야 합니다:

1. 계열(vendor/family) 단위 경고를 추가 — 계열 판정을 무엇으로 할지가 문제
   (모델 ID 접두사? 프로바이더 이름? 수동 매핑표?). 오탐이 나면 정상 설정이
   막힙니다.
2. 문서에 한계로만 명시하고 검사는 ID 단위로 유지 — 구현은 안 늘지만 사용자가
   구멍을 모르고 지나갈 수 있습니다.

**막지 않기로 했습니다.** 이유 둘:

1. 프로바이더 키가 하나뿐이면 계열 경고가 **매 실행마다** 뜹니다. 기본 설정에서
   뜨는 경고는 노이즈이고, 노이즈가 되면 진짜 경고도 같이 무시됩니다.
2. 계열 판정 기준 자체가 애매합니다. 접두사로 자를지 벤더로 자를지, OpenRouter
   경유는 어떻게 볼지. 오탐이 나면 정상 설정이 막힙니다.

**대신 관측 가능하게 두었습니다.** 실행 끝에 사실 한 줄이 찍히고(경고 아님),
`debates` 테이블에 `judge_family` · `participant_families` ·
`judge_shares_family` 가 기록됩니다:

```
judge: gemini 계열 (참가자와 동일; 참가자 계열: gemini)
```

**두 번째 프로바이더 키를 붙이면 해소됩니다.** 그때 같은 계열일 때와 다를 때의
점수 분포를 실측으로 비교할 수 있습니다 — 추측으로 막는 것보다 낫습니다.
