# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 빌드

```bash
# 의존성 설치
brew install cmake sdl2

# 빌드 (whisper.cpp가 ../whisper.cpp에 있어야 함)
mkdir -p build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)

# whisper.cpp 경로 지정 빌드
cmake -DWHISPER_CPP_DIR=/path/to/whisper.cpp ..
make -j$(sysctl -n hw.ncpu)
```

빌드 결과물: `build/bin/live-subtitle`

테스트 프레임워크와 린터는 설정되어 있지 않음.

## 아키텍처

실시간 음성 인식 자막 시스템. 마이크 입력을 whisper.cpp로 처리하여 브라우저에 실시간 자막을 표시.

```
마이크(SDL2) → whisper.cpp 추론 → HTTP/SSE 서버 → 브라우저
```

### 스레딩 모델
- **메인 스레드**: 오디오 캡처 → whisper.cpp 추론 → 공유 상태 업데이트
- **서버 스레드**: HTTP 서버 (cpp-httplib), SSE 스트리밍
- `subtitle_state` 구조체를 mutex + condition_variable로 스레드 간 동기화

### 핵심 파일
- `src/main.cpp` — 전체 애플리케이션 로직 (오디오 캡처, 추론, HTTP 서버, 웹 UI 임베딩)
- `web/index.html` — 자막 표시 웹 UI (참고용, 실제로는 main.cpp에 문자열로 임베딩됨)
- `third_party/httplib.h` — cpp-httplib v0.20.0 (단일 헤더 HTTP 서버 라이브러리)
- `CMakeLists.txt` — C++17, whisper.cpp를 서브디렉토리로 포함

### 외부 의존성
- **whisper.cpp**: 음성 인식 엔진 (CMake 서브프로젝트, 기본 경로 `../whisper.cpp`)
- **SDL2**: 오디오 캡처 (시스템 패키지)
- **cpp-httplib**: HTTP/SSE 서버 (`third_party/httplib.h`에 포함)

### 환각 방지 메커니즘
1. 에너지 기반 VAD로 무음 구간 건너뜀 (`--vad-thold`)
2. 동일 텍스트 3회 이상 반복 시 출력 생략

### 웹 UI
- `web/index.html`은 참고용이며, 실제 서빙되는 HTML은 `main.cpp`의 `INDEX_HTML` 상수에 임베딩됨
- 웹 UI 수정 시 `main.cpp`의 `INDEX_HTML` 문자열을 직접 수정해야 함

## 코드 작성 워크플로우

모든 코드를 Opus 모델로 작성한 후 반드시 `code-simplifier` 에이전트를 실행하여 코드를 점검할 것:
- 구현이 완료된 코드는 즉시 code-simplifier 에이전트로 점검
- 더 이상 개선사항이 없을 때까지 반복 개선
- code-simplifier는 기능을 변경하지 않고 명확성, 일관성, 유지보수성만 개선
