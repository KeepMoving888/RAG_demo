"""
鍚戦噺瀛樺偍妯″潡锛氫娇鐢?Chroma 杩涜鍚戦噺瀛樺偍鍜屾绱?鏀寔鏈湴妯″瀷鍜屽湪绾垮祵鍏?API
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple
from langchain_core.documents import Document
import numpy as np
import config
import re
from pathlib import Path


class EmbeddingClient:
    """宓屽叆妯″瀷瀹㈡埛绔紝鏀寔鏈湴妯″瀷鍜屽湪绾?API"""

    def __init__(self):
        self.model_name = config.EMBEDDING_MODEL
        self._local_model = None
        self._api_client = None
        self._use_api = self._is_api_mode()
        print(f"宓屽叆妯″紡锛歿'鍦ㄧ嚎 API' if self._use_api else '鏈湴妯″瀷'}")

    def _is_api_mode(self) -> bool:
        if config.IS_STREAMLIT_CLOUD:
            return True
        model_path = Path(self.model_name)
        if model_path.exists():
            return False
        api_keywords = ["text-embedding", "embedding-"]
        return any(kw in self.model_name.lower() for kw in api_keywords)

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"姝ｅ湪鍔犺浇鏈湴宓屽叆妯″瀷锛歿self.model_name}")
            self._local_model = SentenceTransformer(self.model_name)
            print("鉁?鏈湴宓屽叆妯″瀷鍔犺浇瀹屾垚")
        return self._local_model

    def _get_api_client(self):
        if self._api_client is None:
            from openai import OpenAI
            api_key = config.EMBEDDING_API_KEY.strip()
            if not api_key:
                api_key = config.LLM_API_KEY.strip()
            if not api_key:
                raise ValueError("璇烽厤缃?EMBEDDING_API_KEY 鎴?LLM_API_KEY")
            self._api_client = OpenAI(
                base_url=config.EMBEDDING_API_BASE,
                api_key=api_key
            )
        return self._api_client

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False, normalize_embeddings: bool = True) -> np.ndarray:
        if self._use_api:
            return self._encode_via_api(texts, batch_size, normalize_embeddings)
        else:
            return self._encode_local(texts, batch_size, show_progress_bar, normalize_embeddings)

    def _encode_local(self, texts: List[str], batch_size: int, show_progress_bar: bool, normalize_embeddings: bool) -> np.ndarray:
        model = self._get_local_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings
        )
        return embeddings

    def _encode_via_api(self, texts: List[str], batch_size: int, normalize_embeddings: bool) -> np.ndarray:
        client = self._get_api_client()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"宓屽叆 API 璋冪敤澶辫触锛歿e}")
                raise

        embeddings = np.array(all_embeddings)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)
        return embeddings

    def encode_single(self, text: str, normalize_embeddings: bool = True) -> List[float]:
        result = self.encode([text], normalize_embeddings=normalize_embeddings)
        return result[0].tolist()


class VectorStore:
    """鍚戦噺瀛樺偍绫?""

    def __init__(self, persist_dir: str, collection_name: str = None):
        self.persist_dir = persist_dir
        if collection_name is None:
            model_name = config.EMBEDDING_MODEL
            if Path(model_name).exists():
                model_tag = Path(model_name).name
            else:
                model_tag = model_name.replace("/", "_").replace("-", "_")
            model_tag = re.sub(r"[^a-zA-Z0-9_]", "_", model_tag).lower()
            if len(model_tag) < 3:
                model_tag = "default_model"
            model_tag = model_tag[:40]
            self.collection_name = f"kb_{model_tag}"
        else:
            self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"姝ｅ湪鍒濆鍖栧祵鍏ユā鍨嬶細{config.EMBEDDING_MODEL}")
        self.embedding_model = EmbeddingClient()
        print("鉁?宓屽叆妯″瀷鍒濆鍖栧畬鎴?)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.metadata.get("chunk_id", f"doc_{i}") for i, doc in enumerate(documents)]

        embeddings = self.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.7
    ) -> List[Tuple[Document, float]]:
        query_embedding = self.embedding_model.encode_single(query, normalize_embeddings=True)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,
            include=["documents", "metadatas", "distances"]
        )

        docs_with_scores = []
        if results['documents'] and results['documents'][0]:
            for i, doc_text in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                score = 1 - distance

                if score >= score_threshold:
                    doc = Document(
                        page_content=doc_text,
                        metadata=results['metadatas'][0][i]
                    )
                    docs_with_scores.append((doc, score))

        return docs_with_scores[:k]

    def get_collection_stats(self) -> Dict:
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_dir": self.persist_dir
        }

    def clear_collection(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


if __name__ == "__main__":
    store = VectorStore(persist_dir="./data/chroma_db")
    print(store.get_collection_stats())
