import sys
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.core.exceptions import CustomException
from backend.core.logger import logging


class ChunkRetriever:
    """
    Lightweight local retriever: TF-IDF + cosine similarity. No external
    embedding API or vector database needed, which keeps this fast, free,
    and dependency-light for a single-document use case like one
    statement at a time. Swap in a proper vector store (e.g. FAISS,
    Chroma, pgvector) if you later need to retrieve across many
    documents at once.
    """

    def __init__(self, chunks: List[str]):
        try:
            self.chunks = chunks
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.chunk_vectors = self.vectorizer.fit_transform(chunks)
        except Exception as e:
            raise CustomException(e, sys)

    def retrieve(self, query: str, top_k: int = 8) -> List[str]:
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]

            top_indices = similarities.argsort()[::-1][:top_k]
            retrieved = [self.chunks[i] for i in top_indices if similarities[i] > 0]

            logging.info(f"Retrieved {len(retrieved)} chunks for query: {query[:60]}...")
            return retrieved

        except Exception as e:
            raise CustomException(e, sys)
