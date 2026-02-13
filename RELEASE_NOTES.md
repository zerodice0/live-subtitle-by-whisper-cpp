# Live Subtitle v1.0.0 Release Notes

Release date: 2026-02-14

## 한국어

### 개요
Live Subtitle 1.0.0은 OBS 크로마키 워크플로우에 맞춘 UI 개선, 음성 인식 안정화, 장치 선택성 개선을 포함한 첫 배포 버전입니다.

### 주요 변경 사항
- OBS 친화 UI
  - 기본 화면을 크로마키용 초록 배경(`\#00FF00`) 기반으로 구성
  - 설정 모드(`?settings=1`)에서 언어/상태 설정 UI 제공
- 소스 언어 제어
  - 소스 언어 선택 API 및 설정 반영
  - `/api/config` 확장 및 `/api/source-languages` 추가
- 오디오 장치 선택 개선
  - `--capture-name` 옵션 추가 (이름 기반 장치 고정 선택)
- 인식 튜닝 옵션 추가
  - `--beam-size` (1~8)
  - `--max-tokens`
  - `--temperature-inc`
- 안정성 개선
  - 중복/반복 텍스트 억제 로직 강화
  - VAD 게이트 및 초기 워밍업 동작 개선

### 지원 플랫폼
- macOS (Apple Silicon)

### 참고
- `--beam-size`는 내부 디코더 제한으로 최대 8까지 지원합니다.

## English

### Overview
Live Subtitle 1.0.0 is the first distribution-ready release, focused on OBS chroma-key workflow improvements, transcription stability, and better capture-device control.

### Highlights
- OBS-friendly UI
  - Default chroma-key green background (`\#00FF00`)
  - Settings mode (`?settings=1`) for language and status controls
- Source language control
  - Source-language configuration support
  - Extended `/api/config` and added `/api/source-languages`
- Better capture device selection
  - Added `--capture-name` for name-based stable device selection
- New transcription tuning options
  - `--beam-size` (1..8)
  - `--max-tokens`
  - `--temperature-inc`
- Stability improvements
  - Stronger duplicate/repetition suppression
  - Improved VAD gating and startup warmup behavior

### Supported Platform
- macOS (Apple Silicon)

### Notes
- `--beam-size` is capped at 8 due to internal decoder limits.
