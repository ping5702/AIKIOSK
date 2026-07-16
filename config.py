import os
from pathlib import Path

# 모델들이 이미 로컬에 캐시되어 있으므로, 매번 허깅페이스 서버에 최신 버전인지
# 확인하러 나가지 않도록 기본값을 오프라인으로 설정한다 (그 확인 요청이 504로
# 응답 없이 걸리는 문제가 있었음). 새 모델을 처음 받아야 할 때는 실행 전에
# 셸에서 `HF_HUB_OFFLINE=0`으로 명시적으로 덮어쓸 것.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"

# 실제 서비스에서는 이 CSV가 현장에서 수집된 장소 정보로 교체될 예정
PLACE_CSV_PATH = RAW_DATA_DIR / "sample_places.csv"

FAISS_INDEX_PATH = INDEX_DIR / "places.index"
METADATA_PATH = INDEX_DIR / "places_metadata.json"

# Ko-SRoBERTa 기반 BiEncoder (문장 임베딩용 sentence-transformers 모델)
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

TOP_K = 3

# 이 값보다 유사도 점수가 낮은 검색 결과는 "관련 정보 없음"으로 처리한다.
# (코사인 유사도 기준: 실제 관련 질문은 0.48~0.6, 무관한 질문은 0.13~0.24로 관측됨)
MIN_SIMILARITY = 0.35

# 검색된 장소 정보를 바탕으로 자연어 답변을 생성하는 로컬 LLM (Ollama)
LLM_MODEL_NAME = "qwen2.5:3b"
OLLAMA_HOST = "http://localhost:11434"

# 답변을 음성으로 합성하는 로컬 TTS 서버 (별도 .venv-tts에서 실행되는 tts_server/server.py)
TTS_SERVER_HOST = "http://localhost:8890"
AUDIO_OUTPUT_DIR = DATA_DIR / "audio"
AUDIO_OUTPUT_PATH = AUDIO_OUTPUT_DIR / "answer.wav"

# 질문 음성을 텍스트로 변환하는 STT 모델 (한국어 파인튜닝된 Whisper-small)
STT_MODEL_NAME = "SungBeom/whisper-small-ko"
STT_DEVICE = "cpu"
STT_TEST_AUDIO_PATH = AUDIO_OUTPUT_DIR / "stt_test_input.wav"
