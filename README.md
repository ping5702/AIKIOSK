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

- Python 3.13
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
setx OLLAMA_HOST "127.0.0.1:11999"
ollama pull qwen2.5:3b

# 3. TTS (Docker, 메인 .venv와 분리된 컨테이너)
docker build -t aikiosk-tts -f tts_server/Dockerfile .
docker run -d -p 8890:8890 --name aikiosk-tts aikiosk-tts

# 4. RAG 인덱스 빌드 (최초 1회 필요, data/index/는 git 미포함)
.venv\Scripts\python build_index.py
```

- `setx`는 영구 환경변수 설정이라 적용하려면 새 터미널을 열어야 합니다.
- 이 값이 필요한 이유: 기본 Ollama 포트(11434)가 이 PC에서 Windows(Hyper-V/WSL2, Docker Desktop)와 충돌해서, `config.py`의 `OLLAMA_HOST`를 `11999`로 옮겨뒀습니다. Ollama도 같은 포트로 맞춰야 합니다.
- 다른 PC에서 11434 포트가 비어있다면, 이 단계 없이 `config.py`를 기본 포트로 되돌려도 됩니다.

## 실행

```powershell
.venv\Scripts\python app.py        # 웹 UI: http://127.0.0.1:7860
.venv\Scripts\python query_test.py # 텍스트 CLI
.venv\Scripts\python mic_test.py   # 마이크 CLI
```

재부팅 후에는 Ollama와 Docker Desktop이 켜져 있는지, `aikiosk-tts` 컨테이너가 실행 중인지(`docker start aikiosk-tts`) 먼저 확인해야 합니다.

## 장소 데이터 수정하기

`data/raw/sample_places.csv`를 수정한 뒤 `build_index.py`를 다시 실행하면 됩니다.

- 필수 컬럼: `id, name, category, description, location`
- 선택 컬럼: `hours`, `phone`
- 필드 안에 쉼표가 있으면 큰따옴표로 감싸야 합니다.
  - 예: `"경영지원 업무 담당 구역 (김다영, 박성우, 장성훈 등)"`
  - 엑셀/구글시트로 저장하면 자동으로 처리됩니다.
- 인덱스를 재빌드했다면 실행 중인 `app.py`/`query_test.py`/`mic_test.py`를 재시작해야 합니다.
  - 인덱스는 프로세스 시작 시 한 번만 메모리에 로드되기 때문입니다.

## 알려진 이슈

- **Windows Smart App Control이 torch DLL을 차단** (`OSError: [WinError 4551]`)
  - 설정 > 개인정보 및 보안 > Windows 보안 > 앱 및 브라우저 컨트롤에서 꺼야 합니다.
  - 한 번 끄면 재설치 전까지 다시 켤 수 없는 단방향 설정입니다.
- **모델 다운로드 시 SSL 인증서 오류** (사내망/방화벽 환경)
  - `pip install pip-system-certs`로 Windows 인증서 저장소를 쓰게 하면 대체로 해결됩니다.
- **모델이 로컬에 캐시돼 있어도 로딩이 느리거나 멈춤**
  - 허깅페이스 서버로 최신 버전을 확인하러 나갔다가 실패하는 경우입니다.
  - `config.py`의 `HF_HUB_OFFLINE=1`이 기본으로 이를 막아줍니다. 새 모델을 처음 받을 때만 이 값을 꺼야 합니다.
- **마이크 필요**: `mic_test.py`/`app.py`는 실제 마이크 장치가 있어야 동작합니다 (`sounddevice` 사용).
- **GPU 미사용**: GPU(RTX 4050, VRAM 6GB 기준 개발)가 있어도 CPU 전용 torch를 설치하므로 STT/임베딩은 전부 CPU로 동작합니다. 응답이 느릴 수 있습니다.
- **Ollama 포트 변경**: 이 PC는 기본 포트(11434)가 충돌해 11999를 씁니다. 다른 PC에서 충돌이 없다면 `config.py`를 기본 포트로 되돌려도 됩니다.

## 프로젝트 구조

| 경로 | 역할 |
|---|---|
| `config.py` | 경로/모델명/서버 주소 등 전체 설정 |
| `src/preprocessing/` | CSV 로드/검증, 임베딩용 텍스트 생성 |
| `src/embedding/`, `src/vectorstore/`, `src/rag/` | 임베딩, FAISS 저장소, 하이브리드 검색 |
| `src/generation/` | Ollama LLM 클라이언트 및 답변 생성 |
| `src/tts/`, `tts_server/` | TTS 클라이언트 및 Docker로 격리된 MeloTTS 서버 |
| `src/stt/` | Silero VAD 녹음기, Whisper STT |
| `src/smalltalk/` | 인사/잡담 즉답 처리 |
| `src/kiosk_pipeline.py` | 위 컴포넌트를 묶은 공용 파이프라인 |
| `app.py` / `query_test.py` / `mic_test.py` | 실행 진입점 (웹 / 텍스트 / 마이크) |
| `build_index.py` | CSV → FAISS 인덱스 빌드 |

자세한 개발 이력과 트러블슈팅은 [plan.md](plan.md) 참고.
