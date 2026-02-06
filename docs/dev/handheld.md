# Handheld

<!-- PROJ_UNDERSTANDING_BEGIN -->

## Project Understanding

### What this project is

- OpenAI와 Google(Gemini) 모델을 통합하여 제공하는 FastAPI 기반의 LLM 게이트웨이 서비스입니다.
- 요청 모델명에 따라 적절한 LLM 공급자(Provider)를 자동으로 선택하는 라우팅 기능을 포함합니다.
- OpenAI 규격의 API를 제공하여 기존 OpenAI 클라이언트와의 호환성을 지향합니다.

### Architecture link

- <!-- PROJ_ARCH_LINK -->docs/dev/architect/architecture_v0.0.0.md

### How to run

- 로컬 실행: `bin/project run` (uvicorn 기반, 8060 포트)
- 컨테이너 빌드: `bin/project build`
- 컨테이너 실행: `bin/project run-container`

### How to test (unit)

- `uv run pytest` (pyproject.toml에 설정된 기본 테스트 실행 방식)

### How to run e2e

- `bin/project test-api` (bin/api_test 스크립트 실행)

### Conventions / gotchas

- 라우팅 로직(`SimpleRouter`)에서 모델명 접두사(`gpt`, `o1`, `gemini` 등)를 기준으로 공급자를 결정합니다.
- `settings.DEFAULT_PROVIDER`의 기본값은 `google`입니다.
- 새로운 모델명이 추가될 경우 `SimpleRouter._select_provider`의 접두사 체크 로직을 업데이트해야 할 수 있습니다.
<!-- PROJ_UNDERSTANDING_END -->

<!-- PROJ_WORKNOTES_BEGIN -->

## Work Notes by Detail

- plan_0001: OpenAI `o3` 모델 지원 추가 (`SimpleRouter`), `bin/project`의 `test-api` 경로 수정 및 컨테이너 실행 시 `OPENAI_API_KEY` 주입 추가.
<!-- PROJ_WORKNOTES_END -->
