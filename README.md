# Live Subtitle

실시간 음성 인식 자막 시스템. 마이크 입력을 [whisper.cpp](https://github.com/ggerganov/whisper.cpp)로 처리하여 브라우저에 실시간 자막을 표시합니다.

```
마이크(SDL2) → whisper.cpp 추론 → HTTP/SSE 서버 → 브라우저
```

## 요구 사항

- macOS (Apple Silicon 지원)
- CMake 3.14 이상
- SDL2
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- Whisper 모델 파일 (`.bin`)

## 설치

### 1. 의존성 설치

```bash
brew install cmake sdl2
```

### 2. whisper.cpp 준비

```bash
cd /path/to/development
git clone https://github.com/ggerganov/whisper.cpp.git
```

### 3. 모델 다운로드

```bash
cd whisper.cpp/models
./download-ggml-model.sh large-v3-turbo
```

사용 가능한 모델 목록:

| 모델 | 크기 | 속도 | 정확도 |
|------|------|------|--------|
| `tiny` | ~75 MB | 가장 빠름 | 낮음 |
| `base` | ~142 MB | 빠름 | 보통 |
| `small` | ~466 MB | 보통 | 좋음 |
| `medium` | ~1.5 GB | 느림 | 높음 |
| `large-v3-turbo` | ~1.6 GB | 보통 | 매우 높음 |

### 4. 빌드

기본 빌드 (whisper.cpp가 `../whisper.cpp`에 있는 경우):

```bash
cd /path/to/live-subtitle
mkdir -p build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

whisper.cpp가 다른 경로에 있는 경우 `-DWHISPER_CPP_DIR`로 지정:

```bash
cmake -DWHISPER_CPP_DIR=/path/to/whisper.cpp ..
make -j$(sysctl -n hw.ncpu)
```

빌드 결과물은 `build/bin/live-subtitle`에 생성됩니다.

## 사용법

### 기본 실행

```bash
./build/bin/live-subtitle --model /path/to/whisper.cpp/models/ggml-large-v3-turbo.bin
```

실행 후 브라우저에서 `http://localhost:8080`을 열면 실시간 자막이 표시됩니다.

### 명령줄 옵션

```
옵션                    설명                          기본값
--model PATH           Whisper 모델 파일 경로         models/ggml-large-v3-turbo.bin
--port N               HTTP 서버 포트                 8080
--language LANG        인식 언어 (ko, en, ja 등)      auto (자동 감지)
--step N               오디오 처리 간격 (ms)           3000
--length N             오디오 버퍼 길이 (ms)           10000
--keep N               이전 오디오 유지 길이 (ms)       200
--threads N            추론 스레드 수                  4
--capture N            오디오 장치 ID                  -1 (기본 장치)
--vad-thold F          음성 감지 에너지 임계값          0.6
--no-gpu               GPU 비활성화
--no-flash-attn        Flash Attention 비활성화
-h, --help             도움말 표시
```

### 사용 예시

한국어 전용으로 실행:

```bash
./build/bin/live-subtitle \
  --model /path/to/models/ggml-large-v3-turbo.bin \
  --language ko
```

특정 마이크 장치 사용 (장치 ID는 실행 시 로그에 표시됨):

```bash
./build/bin/live-subtitle \
  --model /path/to/models/ggml-large-v3-turbo.bin \
  --capture 2
```

다른 포트에서 실행:

```bash
./build/bin/live-subtitle \
  --model /path/to/models/ggml-large-v3-turbo.bin \
  --port 3000
```

빠른 응답을 위해 처리 간격 줄이기 (CPU 부하 증가):

```bash
./build/bin/live-subtitle \
  --model /path/to/models/ggml-large-v3-turbo.bin \
  --step 1500
```

### 종료

`Ctrl+C`로 종료합니다.

## 프로젝트 구조

```
live-subtitle/
├── CMakeLists.txt      # 빌드 설정
├── src/
│   └── main.cpp        # 메인 소스 (오디오 캡처 + 추론 + HTTP 서버)
├── web/
│   └── index.html      # 자막 표시 웹 UI (참고용, 바이너리에 임베딩됨)
├── third_party/
│   └── httplib.h       # cpp-httplib (HTTP 서버 라이브러리)
└── build/              # 빌드 디렉토리
```

## 동작 원리

1. SDL2로 마이크에서 오디오를 실시간 캡처
2. 설정된 간격(`--step`)마다 오디오 데이터를 whisper.cpp에 전달
3. 에너지 기반 VAD로 무음 구간은 건너뜀 (환각 방지)
4. 동일 텍스트 3회 이상 반복 시 출력 생략 (환각 방지)
5. 인식 결과를 SSE(Server-Sent Events)로 연결된 브라우저에 실시간 전송
6. 브라우저에서 자막 스타일로 텍스트 표시, 5초간 입력 없으면 페이드 처리
