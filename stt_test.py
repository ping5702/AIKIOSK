import config
from src.embedding.bi_encoder import BiEncoder
from src.rag.retriever import PlaceRetriever
from src.stt.whisper_stt import WhisperSTT
from src.vectorstore.faiss_store import FaissPlaceStore


def main():
    stt = WhisperSTT(model_name=config.STT_MODEL_NAME, device=config.STT_DEVICE)
    text = stt.transcribe(str(config.STT_TEST_AUDIO_PATH))
    print(f"[STT 인식 결과] {text}")

    encoder = BiEncoder(config.EMBEDDING_MODEL_NAME)
    store = FaissPlaceStore.load(config.FAISS_INDEX_PATH, config.METADATA_PATH)
    retriever = PlaceRetriever(encoder, store, top_k=config.TOP_K, min_similarity=config.MIN_SIMILARITY)
    results = retriever.search(text)
    print(retriever.format_answer(results))


if __name__ == "__main__":
    main()
