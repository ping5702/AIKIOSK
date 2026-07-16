import config
from src.embedding.bi_encoder import BiEncoder
from src.preprocessing.place_preprocessor import preprocess_places
from src.vectorstore.faiss_store import FaissPlaceStore


def main():
    records = preprocess_places(config.PLACE_CSV_PATH)
    texts = [r["embedding_text"] for r in records]

    encoder = BiEncoder(config.EMBEDDING_MODEL_NAME)
    embeddings = encoder.encode(texts)

    store = FaissPlaceStore(dim=embeddings.shape[1])
    store.add(embeddings, records)
    store.save(config.FAISS_INDEX_PATH, config.METADATA_PATH)

    print(f"{len(records)}개의 장소 정보를 인덱싱했습니다. -> {config.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
