import winsound
from pathlib import Path
from typing import Optional

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
            self.encoder,
            self.store,
            top_k=config.TOP_K,
            min_similarity=config.MIN_SIMILARITY,
            bm25_weight=config.BM25_WEIGHT,
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

    def synthesize_to_file(self, text: str) -> Optional[Path]:
        """답변을 음성으로 합성해 파일로 저장하고 그 경로를 반환한다.
        TTS 서버에 연결할 수 없으면 None을 반환한다 (재생은 호출한 쪽이 결정)."""
        try:
            audio_bytes = self.tts_client.synthesize(text)
        except requests.exceptions.RequestException:
            return None

        config.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.AUDIO_OUTPUT_PATH.write_bytes(audio_bytes)
        return config.AUDIO_OUTPUT_PATH

    def speak(self, text: str) -> None:
        """CLI 스크립트용: 답변을 출력하고, 합성한 음성을 서버 스피커로 바로 재생한다."""
        print(text)
        audio_path = self.synthesize_to_file(text)
        if audio_path is None:
            print("(TTS 서버에 연결할 수 없어 텍스트만 출력합니다)")
            return

        print(f"(음성 저장됨: {audio_path})")
        winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
