"""
RAG核心引擎：整合检索和生成
"""
from typing import List, Dict, Tuple
from pathlib import Path
from collections import Counter
import json
import re

import jieba
from langchain.schema import Document
from openai import OpenAI

from document_parser import DocumentProcessor
from vector_store import VectorStore
import config


class RAGEngine:
    """RAG引擎类"""

    def __init__(self):
        self.doc_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.vector_store = VectorStore(persist_dir=str(config.CHROMA_DIR))

        api_key = (config.LLM_API_KEY or "").strip() or "not-needed"
        self.llm_client = OpenAI(base_url=config.LLM_API_BASE, api_key=api_key)

        self.system_prompt = """你是由[HERO]开发的专业企业智能问答助手，专注于为用户提供准确、可靠的企业相关信息和服务。

# 核心原则
1. 准确性优先：只回答有确切依据的问题，不确定时明确告知用户
2. 信息边界：不编造、不推测企业内部未公开信息
3. 专业态度：保持礼貌、专业、中立的沟通风格
4. 安全合规：不泄露敏感信息，遵守数据隐私保护规定

# 回复规范
1. 结构化呈现信息（分点/分段）
2. 标注信息来源或时效性
3. 复杂问题提供后续行动建议
4. 无法回答时提供替代方案

如果上下文没有答案，请直接回答："未找到相关文档信息"。

【上下文信息】
{context}

【用户问题】
{question}

请回答："""

    def ingest_documents(self, file_paths: List[str]) -> Dict:
        """导入文档到知识库"""
        all_docs = []
        stats = {"success": 0, "failed": 0, "total_chunks": 0}

        for file_path in file_paths:
            try:
                docs = self.doc_processor.process_file(file_path)
                all_docs.extend(docs)
                stats["success"] += 1
                stats["total_chunks"] += len(docs)
            except Exception as e:
                stats["failed"] += 1
                print(f"处理失败 {file_path}: {e}")

        if all_docs:
            self.vector_store.add_documents(all_docs)

        stats["vector_count"] = self.vector_store.get_collection_stats()["document_count"]
        return stats

    def ingest_directory(self, dir_path: Path) -> Dict:
        """导入目录下的文档到知识库"""
        if not dir_path.exists():
            return {"success": 0, "failed": 0, "total_chunks": 0, "vector_count": self.vector_store.get_collection_stats()["document_count"]}

        files = []
        for p in dir_path.iterdir():
            if p.is_file() and p.suffix.lower() in config.SUPPORTED_DOC_EXTS:
                files.append(str(p.resolve()))

        return self.ingest_documents(files)

    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        k = k or config.TOP_K
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            score_threshold=config.SIMILARITY_THRESHOLD,
        )

    def _extract_keywords(self, query: str, top_k: int = 5) -> List[str]:
        words = jieba.lcut(query)
        stop_words = {
            "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
            "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这",
        }
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        if not keywords:
            keywords = words
        keyword_freq = Counter(keywords)
        return [kw for kw, _ in keyword_freq.most_common(top_k)]

    def _keyword_search(self, keywords: List[str], k: int = 5) -> List[Tuple[Document, float]]:
        all_docs = []

        for keyword in keywords:
            keyword_embedding = self.vector_store.embedding_model.encode(
                [keyword],
                normalize_embeddings=True,
            )[0].tolist()

            results = self.vector_store.collection.query(
                query_embeddings=[keyword_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            if results["documents"] and results["documents"][0]:
                for i, doc_text in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i]
                    score = 1 - distance
                    doc = Document(page_content=doc_text, metadata=results["metadatas"][0][i])
                    all_docs.append((doc, score))

        all_docs.sort(key=lambda x: x[1], reverse=True)

        seen_chunks = set()
        unique_docs = []
        for doc, score in all_docs:
            chunk_id = doc.metadata.get("chunk_id", "")
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_docs.append((doc, score))
                if len(unique_docs) >= k:
                    break

        return unique_docs

    def generate_answer(self, query: str, context_docs: List[Tuple[Document, float]]) -> Dict:
        context_parts = []
        sources = []

        for doc, score in context_docs:
            context_parts.append(f"[相似度：{score:.2f}] {doc.page_content}")
            sources.append(
                {
                    "file": doc.metadata.get("source", "未知"),
                    "chunk_id": doc.metadata.get("chunk_id", "未知"),
                    "score": round(score, 3),
                }
            )

        context = "\n\n".join(context_parts)
        prompt = self.system_prompt.format(context=context, question=query)

        try:
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一位专业的企业知识助手。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            hallucination_check = self._check_hallucination(answer, context)

            return {
                "answer": answer,
                "sources": sources,
                "context_used": len(context_docs),
                "hallucination_risk": hallucination_check,
                "query": query,
            }
        except Exception as e:
            error_text = str(e)
            return {
                "answer": f"生成答案时出错：{error_text}",
                "sources": sources,
                "context_used": len(context_docs),
                "query": query,
                "error": error_text,
            }

    def _check_hallucination(self, answer: str, context: str) -> Dict:
        honest_phrases = ["无法回答", "没有相关信息", "不清楚", "不知道"]
        is_honest = any(phrase in answer for phrase in honest_phrases)
        has_citation = bool(re.search(r"\[来源[：:]", answer))
        return {
            "is_honest": is_honest,
            "has_citation": has_citation,
            "risk_level": "low" if (is_honest or has_citation) else "medium",
        }

    def query(self, question: str, with_sources: bool = True) -> Dict:
        context_docs = self.retrieve(question)

        if not context_docs:
            keywords = self._extract_keywords(question)
            if keywords:
                context_docs = self._keyword_search(keywords, k=config.TOP_K)

        if not context_docs:
            return {
                "answer": "未在知识库中找到相关信息，请尝试补充文档或换一种问法。",
                "sources": [],
                "context_used": 0,
                "query": question,
            }

        result = self.generate_answer(question, context_docs)
        if not with_sources:
            result.pop("sources", None)
        return result

    def get_stats(self) -> Dict:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "config": {
                "chunk_size": config.CHUNK_SIZE,
                "top_k": config.TOP_K,
                "threshold": config.SIMILARITY_THRESHOLD,
            },
        }


rag_engine = RAGEngine()


if __name__ == "__main__":
    engine = RAGEngine()
    result = engine.query("公司的年假政策是什么？")
    print(json.dumps(result, ensure_ascii=False, indent=2))
