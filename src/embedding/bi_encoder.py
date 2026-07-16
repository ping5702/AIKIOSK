from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class BiEncoder:
    """Ko-SRoBERTa 기반 BiEncoder로 텍스트를 문장 임베딩 벡터로 변환한다."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
