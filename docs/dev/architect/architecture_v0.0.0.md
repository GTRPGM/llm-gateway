# Architecture v0.0.0

## Summary

- initial architecture

## Context

- 다중 LLM 공급자(OpenAI, Gemini 등)를 단일 인터페이스로 통합 제공하는 게이트웨이 서비스입니다.
- OpenAI SDK 규격의 API를 노출하여 기존 클라이언트의 변경을 최소화합니다.

## System overview

- **Framework**: FastAPI
- **Configuration**: Pydantic Settings (`.env` 지원)
- **Core Components**:
  - `LLMEngine`: 서비스 비즈니스 로직 진입점
  - `SimpleRouter`: 모델명 접두사 기반 공급자 선택
  - `Providers`: 각 LLM API 연동 (OpenAI, Gemini)

## Data flow

1. **Client** -> HTTP POST `/api/v1/chat/completions`
2. **FastAPI Router** -> `LLMEngine.chat()` 호출
3. **LLMEngine** -> `SimpleRouter.route_chat()` 호출
4. **SimpleRouter** -> 모델명 분석 후 `OpenAIProvider` 또는 `GeminiProvider` 선택
5. **Provider** -> 외부 LLM API 호출 및 응답 변환
6. **Response** -> 클라이언트에게 전달 (Stream/Non-stream 지원)

## Decisions

- Decision: 모델명 접두사 기반 라우팅
- Reason: 구현이 단순하며, 신규 모델 추가 시에도 접두사 매칭만으로 대응 가능
- Impact: 라우팅 로직(`SimpleRouter`)을 정기적으로 업데이트해야 함

## Compatibility / migration notes

- TBD
