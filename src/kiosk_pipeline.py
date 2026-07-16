import winsound

import requests

import config
from src.embedding.bi_encoder import BiEncoder
from src.generation.answer_generator import AnswerGenerator
from src.generation.llm_client import OllamaClient
from src.rag.retriever import PlaceRetriever
from src.smalltalk.smalltalk_handler import match_smalltalk
from src.tts.tts_client import MeloTTSClient
from src.vectorstore.faiss_store import FaissPlaceStore


class KioskPipeline:
    """스몰토크 → RAG 검색 → LLM 답변 생성 → TTS 출력까지, 텍스트/음성 입력 어느 쪽에서든
    공통으로 쓰는 응답 파이프라인."""

    def __init__(self):
        self.encoder = BiEncoder(config.EMBEDDING_MODEL_NAME)
        self.store = FaissPlaceStore.load(config.FAISS_INDEX_PATH, config.METADATA_PATH)
        self.retriever = PlaceRetriever(
            self.encoder, self.store, top_k=config.TOP_K, min_similarity=config.MIN_SIMILARITY
        )
        self.llm_client = OllamaClient(model=config.LLM_MODEL_NAME, host=config.OLLAMA_HOST)
        self.answer_generator = AnswerGenerator(self.llm_client)
        self.tts_client = MeloTTSClient(host=config.TTS_SERVER_HOST)

    def answer(self, query: str) -> str:
        smalltalk_response = match_smalltalk(query)
        if smalltalk_response is not None:
            return smalltalk_response

        results = self.retriever.search(query)
        return self.answer_generator.generate(query, results)

    def speak(self, text: str) -> None:
        print(text)
        try:
            audio_bytes = self.tts_client.synthesize(text)
            config.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            config.AUDIO_OUTPUT_PATH.write_bytes(audio_bytes)
            print(f"(음성 저장됨: {config.AUDIO_OUTPUT_PATH})")
            winsound.PlaySound(str(config.AUDIO_OUTPUT_PATH), winsound.SND_FILENAME)
        except requests.exceptions.RequestException:
            print("(TTS 서버에 연결할 수 없어 텍스트만 출력합니다)")
