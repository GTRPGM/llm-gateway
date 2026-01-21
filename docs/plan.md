# LLM Gateway 구축 계획

## 1. 개요

이 프로젝트는 **LLM 기반 TRPG 진행 시스템**의 핵심 인프라 스트럭처인 **LLM Gateway**를 구축하기 위한 설계 및 구현 계획서이다.

본 서비스는 GM Orchestrator, NPC AI, Rule Manager 등 다양한 내부 서비스들로부터 발생하는 자연어 생성(LLM) 요청을 중앙에서 수신하고 처리한다. 이를 통해 각 서비스는 특정 LLM 모델의 API 명세나 변경 사항에 종속되지 않고, **추상화된 인터페이스**를 통해 AI 기능을 활용할 수 있다.

## 2. 핵심 목표

1. **단일 진입점 (Single Entry Point)**
    - 시스템 내 모든 LLM 호출을 통합 관리하여 보안, 인증, 로깅을 중앙화한다.
2. **모델 추상화 (Model Abstraction)**
    - 클라이언트는 "OpenAI", "Google" 등의 구체적 Provider를 몰라도, 기능적 요구사항(예: "창의적 서술", "논리적 판정")에 맞춰 요청할 수 있다.
    - 내부적으로 `gpt-4`, `gemini-pro`, `claude-3` 등 최적의 모델로 라우팅한다.
3. **관측성 (Observability)**
    - 모든 프롬프트와 생성 결과를 추적(Tracing)한다 (예: LangSmith 활용).
    - 비용, 지연 시간(Latency), 에러율을 모니터링한다.
4. **안정성 및 제어 (Reliability & Control)**
    - Rate Limit 관리, 자동 재시도(Retry), 장애 시 Fallback 처리.
    - 개발/운영 환경에 따른 유연한 모델 스위칭.

## 3. 기술 스택

- **Language**: Python 3.11+
- **Web Framework**: FastAPI (비동기 처리, 고성능)
- **Dependency Management**: `uv` 또는 `poetry` (표준 `pyproject.toml` 기반)
- **Libraries**:
  - `pydantic`, `pydantic-settings`: 데이터 검증 및 환경 설정
  - `httpx`: 비동기 HTTP 클라이언트
  - `langsmith`: LLM Tracing 및 평가
  - `openai`, `google-generativeai`: Vendor SDK (필요시)
- **Deployment**: Docker, Uvicorn

## 4. 아키텍처 및 디렉토리 구조

소스 코드는 `src` 레이아웃 패턴을 따른다.

### 4.1 디렉토리 구조 (Src Layout)

```bash
llm-router/
├── src/
│   └── llm_router/
│       ├── __init__.py
│       ├── main.py              # FastAPI 앱 엔트리포인트
│       ├── api/                 # API 라우트 핸들러
│       │   ├── __init__.py
│       │   └── v1/
│       ├── core/                # 핵심 설정, 로깅, 예외 처리
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── exceptions.py
│       ├── schemas/             # Pydantic 모델 (Request/Response DTO)
│       │   ├── __init__.py
│       │   └── chat.py
│       └── services/            # 비즈니스 로직 및 Provider 어댑터
│           ├── __init__.py
│           ├── router.py        # 모델 라우팅 로직
│           └── providers/       # LLM 벤더별 구현
│               ├── base.py
│               ├── openai.py
│               └── gemini.py
├── tests/                       # Pytest 테스트 코드
├── .env                         # 로컬 환경 변수
├── pyproject.toml
└── README.md
```

### 4.2 API 명세 (Draft)

- **POST `/v1/chat/completions`**
  - 표준 OpenAI Chat Completion 포맷을 준수하되, 커스텀 필드를 확장 지원.
  - 요청 Body 예시:

    ```json
    {
      "model": "high-reasoning", // 논리적 추론용 모델 별칭
      "messages": [{ "role": "user", "content": "..." }],
      "temperature": 0.7
    }
    ```

## 5. 개발 로드맵

### Phase 1: 기본 환경 및 스캐폴딩 (완료)

- [x] `pyproject.toml` 의존성 설정 완료.
- [x] `src` 디렉토리 구조 생성 (`src/llm_gateway`).
- [x] `src/llm_gateway/main.py` 헬스 체크 엔드포인트 구현 완료.
- [x] 환경 변수 관리(`pydantic-settings`) 설정 완료.

### Phase 2: LLM Provider 연동 (진행 예정)

- [ ] Provider 추상 클래스(`BaseLLMProvider`) 정의.
- [ ] Google Gemini Provider 구현 및 연동.
- [ ] OpenAI Provider 구현 및 연동.

### Phase 3: 라우팅 및 서비스 로직

- [ ] 모델 별칭(Alias) 시스템 구현 (예: `creative` -> `gpt-4`, `fast` -> `gemini-flash`).
- [ ] `/v1/chat/completions` 엔드포인트 구현.
- [ ] 스트리밍(Streaming) 응답 지원.

### Phase 4: 관측성 및 안정화

- [ ] LangSmith Tracing 연동.
- [ ] 에러 핸들링 및 재시도(Retry) 미들웨어 적용.
- [ ] Dockerfile 작성 및 배포 구성.
