AI안내 키오스크 작업계획

## 프로젝트 목표
목표 : 장소의 정보를 입력받아 사용자가 KIOSK에게 물어봤을 경우 STT를 통해 사내 정보를 RAG를 통해 검색하여 정보를 받아와 다시 TTS를 통해 사용자에게 정보를 전달해주는 키오스크를 개발할 예정

## 개발시 참고
단계별로 테스트를 해가면서 구현할 계획이라 구현하라고 해놓은 곳 까지만 우선적으로 구현하고 추후에 구현할 수 있도록 위치만 잡아주는 방식의 코딩이 필요

## 현재 구현하여 테스트 필요
1. RAG 구현하기
    1. TiRoBERTa BiEncoder Model 사용하여 워드 임베딩 진행
    2. 장소 정보 전처리 → 아직 데이터가 없기 때문에 추후 CSV 형태 혹은 비슷한 포맷으로 입력될 예정임
    3. 임베딩된 데이터를 FAISS  DB에 저장
2. TEXT로  데이터 가져오기
    1. 구현된 DB에서 텍스트를 통하여 정보 조회
    2. 조회된 정보를 안내 형식으로 TEXT형태로 출력
    3. 최소 유사도 threshold 적용 → 관련 없는 질문은 "관련 정보 없음"으로 응답
3. 스몰토크(인사/잡담) 레이어 추가하기
    1. 장소 검색(RAG)과 분리된 별도 모듈(src/smalltalk)로 인사/잡담 의도를 먼저 판별
    2. 매칭되면 고정 응답 반환, 매칭 안되면 기존 RAG 장소 검색으로 진행
4. LLM 연동하여 자연어 답변 생성하기
    1. 로컬 Ollama로 Qwen2.5 3B 서빙 (GPU VRAM 2GB 한계로 CPU 추론)
    2. 검색된 장소 정보(top-k)를 컨텍스트로 넣어 LLM이 답변 생성 (src/generation)
    3. 기존 고정 템플릿(format_answer)은 폴백/참고용으로 유지
5. TTS 연동하여 답변을 음성으로 출력하기 (완료)
    1. 모델: MeloTTS (한국어 지원, MIT 라이선스, CPU real-time)
    2. Windows 네이티브 설치는 fugashi/tokenizers가 Windows용 사전빌드가 없고
       MSVC/Rust 컴파일러도 없어서 실패 → Docker로 전환 (MeloTTS 공식 문서 권장 방식)
    3. WSL2 + Docker Desktop 설치(사용자가 관리자 권한으로 직접 진행) 후,
       tts_server/Dockerfile로 MeloTTS를 컨테이너에 빌드
       - CPU 전용 torch로 설치해 불필요한 CUDA 의존성 제거
       - melotts 자체 requirements.txt를 통째로 풀면 cached_path(→boto3/
         google-cloud-storage)와 gradio 조합에서 pip이 수십 분씩 backtracking에
         빠짐 → melotts는 --no-deps로 설치하고, 한국어 합성에 실제 필요한 패키지만
         직접 골라 설치. gradio(웹UI, 우리는 안 씀)는 완전히 제외
       - 단, cached_path가 안 쓰는 GCS/S3 클라이언트도 모듈 import 시점에 무조건
         import하므로 boto3/google-cloud-storage는 설치가 필요함 → 다른 패키지와
         분리해서 독립적으로 설치(그래야 backtracking 없이 빠르게 끝남)
    4. tts_server/server.py: MeloTTS 모델을 상주시키는 Flask 서버 (텍스트 → wav),
       컨테이너 안에서 0.0.0.0:8890으로 서빙. Ollama를 별도 프로세스로 분리해
       LLM을 연동한 것과 동일한 패턴
    5. src/tts/tts_client.py: 메인 앱에서 이 서버를 requests로 호출 (OllamaClient와
       동일 구조), query_test.py에 연결 완료
    6. 실행: `docker run -d -p 8890:8890 --name aikiosk-tts aikiosk-tts`
       (이미지가 이미 빌드되어 있으면 컨테이너만 재시작하면 됨. 재부팅 후에는
       Docker Desktop과 Ollama 앱을 다시 켜줘야 함 — 둘 다 자동 시작 등록은 되어있음)
    7. 질문→RAG→LLM→TTS 전체 파이프라인 실제 동작 확인 완료
       (예: "카페테리아 전화번호는?" → 자연어 답변 생성 → wav 파일 저장 성공)

6. STT와 RAG 연동하기 (1단계 완료 — 마이크 실시간 입력은 미구현)
    1. 모델: SungBeom/whisper-small-ko (whisper-small 기반 한국어 파인튜닝,
       AI Hub 고객응대음성 등으로 학습해 키오스크 도메인과 잘 맞음, WER 9.48)
       - 대안으로 whisper-large-v3-turbo(809M)도 검토했으나, 우리 하드웨어(VRAM 2GB,
         CPU 추론 위주)엔 whisper-small 기반(244M)이 훨씬 가벼워 더 적합하다고 판단
    2. src/stt/whisper_stt.py: `transformers`의 ASR pipeline으로 오디오를 텍스트로
       변환. soundfile로 wav를 읽고 librosa로 16kHz mono로 리샘플링 후 pipeline에
       raw array로 전달 (ffmpeg 의존성 회피)
    3. 1단계 검증 방법: 실제 마이크 대신, 이미 구현된 TTS 서버로 테스트 문장을
       음성 합성 → 그 wav를 STT로 다시 텍스트화하는 라운드트립으로 테스트
       (stt_test.py). "지하 주차장이 어디에 있나요?" → TTS → STT →
       "지하 주차장이 어디에 있나요" (물음표만 소실, 완벽히 일치) → retriever.search()에
       그대로 넣어도 정상적으로 "지하 주차장" 정보를 검색함 → STT 출력이 기존 RAG
       파이프라인과 별도 처리 없이 바로 호환됨을 확인
    4. 마이크(DeepHearing SDK) 실시간 입력 연동은 아직 미구현. DeepHearing은
       브라우저(JS)/Android SDK만 공식 지원하고 Windows/Python 네이티브 SDK는
       문서상 확인 안 됨 → 브라우저에서 AudioWorklet으로 원시 Float16kHz mono
       PCM을 뽑아 WebSocket으로 Python 서버에 스트리밍하는 방식을 검토 중
       (MediaRecorder로 녹음하면 WebM/Opus로 압축되어 버려서 피해야 함).
       DeepHearing 쪽에 Windows 네이티브 연동 경로가 있는지 별도 문의 필요
    5. VAD(Voice Activity Detection) 구현 완료 — DeepHearing과 무관하게 먼저
       진행 가능한 부분이라 일반 마이크로 우선 구현
       - 최초 구현은 webrtcvad(에너지 기반)였으나(Windows 사전빌드 없어 대체
         `webrtcvad-wheels` 포크 사용), 노이즈 환경에서 발화 구간이 잘못 잘려
         whisper가 환각을 일으키는 문제가 있어 **Silero VAD(신경망 기반)로 교체**
         (아래 8번 항목 참고)
       - src/stt/vad_recorder.py: `sounddevice`로 마이크 입력을 16kHz mono, 512
         샘플(32ms) 단위로 캡처, Silero VAD로 발화 감지. 말이 시작되면 녹음 시작,
         이후 1초 연속 무음이면 자동으로 녹음 종료
       - src/stt/whisper_stt.py에 transcribe_array 추가: 파일 저장 없이 VAD가 뽑은
         numpy 배열을 바로 STT에 전달 가능하도록 리팩터링 (transcribe()는 이제
         transcribe_array()를 호출하는 얇은 래퍼)
       - mic_test.py: 마이크 → VAD → STT → RAG 검색까지 실제 이어지는 대화형
         테스트 스크립트. 단, 실제 육성 입력 테스트는 사용자가 직접 실행해서
         확인해야 함 (에이전트는 마이크 입력을 시뮬레이션할 수 없음)
       - **트러블슈팅**: STT 모델 로딩이 몇 분씩 멈추는 현상 발생 → 원인은
         Whisper 모델이 이미 캐시돼 있어도 `transformers`가 매번 허깅페이스
         서버에 최신 버전인지 HEAD 요청으로 확인하러 나가는데, 이 요청이 504로
         응답 없이 걸림(Docker/WSL2 설치 이후 네트워크 경로 이슈로 추정).
         → config.py에 `os.environ.setdefault("HF_HUB_OFFLINE", "1")`을 추가해
         기본값을 오프라인으로 고정 (모델들이 이미 다 로컬에 캐시돼 있어서 오프라인
         모드로도 전혀 문제없이 동작 확인함). 새 모델을 처음 받을 때만 실행 전
         셸에서 `HF_HUB_OFFLINE=0`으로 명시적으로 덮어쓸 것
    6. 마이크 입력을 전체 파이프라인(스몰토크→RAG→LLM→TTS)에 연결 완료.
       딥히어링 연동 전이라도 먼저 진행 — 지금 상호작용이 "녹음 → 답변 재생 →
       다음 질문 대기" 완전 턴 기반 구조라 에코 제거(AEC) 없이도 TTS 소리를
       마이크가 다시 주워듣는 문제가 없어서, 딥히어링은 나중에 마이크 입력
       소스만 교체하면 되는 구조로 남겨둠
       - src/kiosk_pipeline.py: KioskPipeline 클래스로 스몰토크/RAG/LLM/TTS 로직을
         공용화 (answer(query), speak(text)). query_test.py(텍스트 입력)와
         mic_test.py(마이크 입력) 둘 다 이 클래스를 공유해서 중복 제거
       - mic_test.py: 마이크→VAD→STT→(공용 파이프라인)→TTS 음성 응답까지 실제 동작
       - 실사용 테스트 중 발견한 버그: VAD가 아주 짧은/애매한 구간만 캡처했을 때
         whisper-small-ko가 입력과 무관한 문장을 "환각"으로 반복 출력하는 현상
         있었음 → mic_test.py에 디버그용 저장(data/audio/last_mic_recording.wav)과
         녹음 길이 출력을 추가해 원인 진단 중

7. 간단한 Web/GUI 화면 구현 (완료 — 버튼 방식)
    1. `gradio`로 [app.py](app.py) 작성: 🎙️ 말하기 버튼 → 대화 내역(gr.Chatbot) +
       상태 텍스트("듣는 중..." → "인식 중..." → "답변 생성 중..." → "완료") +
       답변 음성 자동 재생(gr.Audio, autoplay). generator 함수로 단계별 진행
       상태를 yield하며 UI를 실시간 갱신
    2. src/kiosk_pipeline.py에 synthesize_to_file(text) 추가: 웹 UI는 서버
       스피커(winsound)로 재생할 필요 없이 브라우저가 재생하므로, 파일 경로만
       반환하는 메서드를 분리 (speak()는 이 메서드를 내부적으로 사용)
    3. **트러블슈팅**: gradio 설치 후(사실은 무관하게 우연히 같은 시점에) torch
       import가 `OSError: [WinError 4551]`로 막힘 → 원인은 Windows 11의
       **Smart App Control**이 서명되지 않은 torch\lib\shm.dll을 새로 차단한 것
       (이벤트 뷰어 Microsoft-Windows-CodeIntegrity/Operational 로그의 Event ID
       3077/3118로 확인). 설정 > 개인정보 및 보안 > Windows 보안 > 앱 및 브라우저
       컨트롤에서 Smart App Control을 꺼서 해결 (주의: 한번 끄면 재설치 없이는
       다시 켤 수 없는 단방향 설정이라 사용자가 직접 끔)
    4. 실행: `python app.py` → http://127.0.0.1:7860 접속. 서버 기동/HTTP 응답은
       확인했으나, 실제 마이크 버튼 클릭 테스트는 사용자가 직접 확인 필요
       (에이전트는 마이크 입력을 시뮬레이션할 수 없음)
    5. 다음 단계(미구현): "상시 대기" 모드 — 버튼 없이 백그라운드 스레드가 계속
       VAD로 마이크를 감시하다 발화 감지 시 자동으로 파이프라인 실행. TTS 재생
       중에는 이 백그라운드 감시를 일시정지해야 함(AEC 없이 자기 목소리를
       다시 듣는 문제 방지)

8. 성능 최적화 검토 및 일부 적용 (STT/VAD, RAG 개선 완료 — LLM 엔진 교체는 보류)
    1. "온프리미스 성능 극대화" 제안(faster-whisper, Silero VAD, BM25 하이브리드+
       Reranker, Ollama→vLLM+FlashAttention-2)을 검토한 결과:
       - **Silero VAD, BM25 하이브리드**: 우리 상황에 실제로 맞아서 적용(아래)
       - **faster-whisper**: 효과는 있으나 SungBeom/whisper-small-ko를 ct2
         포맷으로 변환하는 단계가 추가로 필요해서 보류
       - **Reranker(bge-m3 등)**: 지금 데이터가 8건뿐이라 FAISS가 이미 전수
         비교(brute-force) 중이라 정확도 개선 효과가 없고, 모델을 하나 더
         돌리는 거라 지연시간엔 오히려 마이너스 → 데이터 늘어난 뒤 재검토
       - **vLLM + FlashAttention-2**: vLLM의 PagedAttention/Continuous batching은
         "동시 다중 사용자 서빙"을 위한 기술인데 우리는 키오스크 1대·순차 요청
         구조라 해당 문제가 없음. 게다가 둘 다 CUDA GPU 전제 기술인데 우리는
         VRAM 2GB라 Ollama도 결국 CPU로 돌렸던 것과 같은 벽에 부딫힘 → 적용 안 함
         (양자화는 이미 Qwen2.5 3B가 Q4_K_M으로 서빙 중이라 별도 조치 불필요)
    2. **Silero VAD로 교체 완료**: src/stt/vad_recorder.py가 `silero-vad`
       패키지의 `load_silero_vad()` 모델을 직접 호출해 청크(512 샘플)별 발화
       확률을 구하는 방식으로 재작성. webrtcvad보다 노이즈 환경에서 더 정확할
       것으로 기대(신경망 기반) — 실사용 노이즈 환경 검증은 사용자가 직접 필요
    3. **BM25 하이브리드 검색 추가 완료**: src/rag/bm25_index.py(`rank-bm25` 사용).
       한글은 조사가 붙어서("3층에") 공백 토큰화만으로는 "3층"과 매칭이 안 되는
       문제가 있어, 2글자 초과 단어는 글자 단위 bigram으로 쪼개는 자체 토크나이저
       사용 (형태소 분석기 없이 간단하게 해결)
       - src/rag/retriever.py: 벡터 유사도(dense)와 BM25 점수를 `(1-w)*dense +
         w*bm25` 로 결합(기본 w=0.3, config.BM25_WEIGHT). "관련 없음" 판단은
         BM25 영향 없이 dense_score만으로 유지(BM25는 키워드 하나만 겹쳐도 점수를
         주기 때문에 무관 질문 필터링 기준으로는 부적합)
       - **개선 확인**: "3층에 있는 시설을 알려줘" 질의가 하이브리드 적용 전엔
         무관한 "종합 안내데스크"를 1위로 반환했는데, 적용 후 정확히
         "대회의실 A"(위치: 3층 301호)를 1위로 반환하도록 개선됨 (실제 회귀
         테스트로 확인)

## 추후 개발 예정 아직 구현 하지 말것
7. DeepHearing SDK로 실시간 마이크 입력 연동하기 → 위 4번 문의 결과 나온 뒤 진행


## 필수 규칙
1. import에 필요한 라이브러리는 항상 requirements에 추가할 것