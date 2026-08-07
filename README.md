# AIKIOSK

장소 정보를 등록해두면, 사용자가 말이나 텍스트로 물어봤을 때 STT → RAG 검색 → LLM 답변 생성 → TTS 음성 안내까지 처리하는 AI 안내 키오스크.

## 아키텍처

```
[입력: 마이크 / 텍스트 / 웹 버튼]
        │
        ▼ (마이크/웹)
   VAD(Silero) 발화 감지 → Whisper(whisper-small-ko)로 텍스트 변환
        │
        ▼
   스몰토크 판별 ("안녕/고마워"면 즉답)
        │ (장소 관련 질문이면)
        ▼
   RAG 검색: BiEncoder(벡터) + BM25(키워드) 하이브리드로 FAISS에서 조회
        │
        ▼
   LLM(Qwen2.5 3B, 로컬 Ollama)이 검색 결과를 근거로 자연어 답변 생성
        │
        ▼
   TTS(MeloTTS, Docker)로 답변 음성 합성 → 재생
```

## 필수 프로그램

- Python 3.12
- Git
- [Ollama](https://ollama.com)
- Docker Desktop (Windows는 WSL2 필요)

## 설치

```powershell
git clone https://github.com/ping5702/AIKIOSK.git
cd AIKIOSK

# 1. 메인 파이썬 가상환경
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. LLM (Ollama)
ollama pull qwen2.5:3b

# 3. TTS (Docker, 메인 .venv와 의존성 충돌 방지를 위해 별도 컨테이너로 격리)
docker build -t aikiosk-tts -f tts_server/Dockerfile .
docker run -d -p 8890:8890 --name aikiosk-tts aikiosk-tts

# 4. RAG 인덱스 빌드 (data/index/는 git에 포함 안 됨, 최초 1회 필요)
.venv\Scripts\python build_index.py
```

## 실행

```powershell
.venv\Scripts\python app.py        # 웹 UI: http://127.0.0.1:7860
.venv\Scripts\python query_test.py # 텍스트 CLI
.venv\Scripts\python mic_test.py   # 마이크 CLI
```

재부팅 후에는 Ollama와 Docker Desktop이 켜져 있는지, `aikiosk-tts` 컨테이너가 실행 중인지(`docker start aikiosk-tts`) 먼저 확인해야 합니다.

## 장소 데이터 수정하기

`data/raw/sample_places.csv`를 수정한 뒤 `build_index.py`를 다시 실행하면 됩니다. CSV 규칙:

- 필수 컬럼: `id, name, category, description, location` (`hours`, `phone`은 선택)
- **필드 안에 쉼표가 있으면 큰따옴표로 감싸야 합니다.** 예: `"경영지원 업무 담당 구역 (김다영, 박성우, 장성훈 등)"`. 엑셀/구글시트로 CSV를 저장하면 자동으로 처리됩니다.
- CSV를 바꾸고 인덱스를 재빌드했다면, **이미 실행 중인 `app.py`/`query_test.py`/`mic_test.py`를 재시작**해야 반영됩니다 (인덱스는 프로세스 시작 시 한 번만 메모리에 로드됨).

## 알려진 이슈

- **Windows Smart App Control이 torch DLL을 차단**하는 경우(`OSError: [WinError 4551]`): 설정 > 개인정보 및 보안 > Windows 보안 > 앱 및 브라우저 컨트롤에서 꺼야 함. (주의: 한 번 끄면 재설치 전까지 다시 켤 수 없는 단방향 설정)
- **모델 다운로드 시 SSL 인증서 오류**(사내망/방화벽 환경): `pip install pip-system-certs`로 Windows 인증서 저장소를 쓰게 하면 대체로 해결됨
- **모델이 이미 로컬에 캐시돼 있어도 로딩이 느리거나 멈춤**: 허깅페이스 서버로 최신 버전 확인 요청이 나가다 실패하는 경우. `config.py`에 `HF_HUB_OFFLINE=1`이 기본으로 켜져 있어 보통 문제 없음 — 새 모델을 처음 받을 때만 이 값을 꺼야 함
- `mic_test.py`/`app.py`는 실제 마이크 장치가 있어야 동작 (`sounddevice` 사용)
- GPU가 약해도(VRAM 2GB 기준 개발) 전부 CPU로 자동 동작 — 별도 설정은 불필요하지만 응답이 느릴 수 있음

## 프로젝트 구조

| 경로 | 역할 |
|---|---|
| `config.py` | 경로/모델명/서버 주소 등 전체 설정 |
| `src/embedding/`, `src/vectorstore/`, `src/rag/` | 임베딩, FAISS 저장소, 하이브리드 검색 |
| `src/generation/` | Ollama LLM 클라이언트 및 답변 생성 |
| `src/tts/`, `tts_server/` | TTS 클라이언트 및 Docker로 격리된 MeloTTS 서버 |
| `src/stt/` | Silero VAD 녹음기, Whisper STT |
| `src/smalltalk/` | 인사/잡담 즉답 처리 |
| `src/kiosk_pipeline.py` | 위 컴포넌트를 묶은 공용 파이프라인 |
| `app.py` / `query_test.py` / `mic_test.py` | 실행 진입점 (웹 / 텍스트 / 마이크) |
| `build_index.py` | CSV → FAISS 인덱스 빌드 |

자세한 개발 이력과 트러블슈팅은 [plan.md](plan.md) 참고.
