"""
RAG核心引擎：整合检索和生成
"""
import os
from typing import List, Dict, Tuple, Optional
from langchain.schema import Document
from openai import OpenAI
import json
import re
import jieba
from collections import Counter

from document_parser import DocumentProcessor
from vector_store import VectorStore
import config


class RAGEngine:
    """RAG引擎类"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore(
            persist_dir=str(config.CHROMA_DIR)
        )
        
        # 初始化LLM客户端（兼容OpenAI API格式）
        api_key = config.LLM_API_KEY.strip()
        if not api_key:
            # 本地无鉴权服务可用占位key；云端服务请通过 .env 配置真实 key
            api_key = "not-needed"

        self.llm_client = OpenAI(
            base_url=config.LLM_API_BASE,
            api_key=api_key
        )
        
        # 系统提示词
        self.system_prompt = """你是由[HERO]开发的专业企业智能问答助手，专注于为用户提供准确、可靠的企业相关信息和服务。


# 核心原则
1. 准确性优先：只回答有确切依据的问题，不确定时明确告知用户
2. 信息边界：不编造、不推测企业内部未公开信息
3. 专业态度：保持礼貌、专业、中立的沟通风格
4. 安全合规：不泄露敏感信息，遵守数据隐私保护规定

# 能力范围
- 企业基本信息查询（公开信息）
- 产品/服务咨询
- 业务流程指引
- 常见问题解答
- 内部政策查询（授权范围内）

# 限制说明
- 不回答涉及商业机密的问题
- 不代做决策或提供法律/财务专业建议
- 超出能力范围时引导用户联系对应部门

# 回复规范
1. 结构化呈现信息（分点/分段）
2. 标注信息来源或时效性
3. 复杂问题提供后续行动建议
4. 无法回答时提供替代方案或联系人

# 语气风格
- 专业但不生硬
- 简洁但不失完整
- 主动确认用户理解程度。
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
        
        # 添加到向量库
        if all_docs:
            self.vector_store.add_documents(all_docs)
        
        stats["vector_count"] = self.vector_store.get_collection_stats()["document_count"]
        return stats
    
    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """检索相关文档"""
        k = k or config.TOP_K
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            score_threshold=config.SIMILARITY_THRESHOLD
        )
    
    def _extract_keywords(self, query: str, top_k: int = 5) -> List[str]:
        """从查询中提取关键词"""
        words = jieba.lcut(query)
        stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        if not keywords:
            keywords = words
        
        keyword_freq = Counter(keywords)
        return [kw for kw, _ in keyword_freq.most_common(top_k)]
    
    def _keyword_search(self, keywords: List[str], k: int = 5) -> List[Tuple[Document, float]]:
        """基于关键词检索文档（使用与主检索相同的嵌入模型，避免维度不一致）"""
        all_docs = []
        
        for keyword in keywords:
            keyword_embedding = self.vector_store.embedding_model.encode(
                [keyword],
                normalize_embeddings=True
            )[0].tolist()

            results = self.vector_store.collection.query(
                query_embeddings=[keyword_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            if results['documents'] and results['documents'][0]:
                for i, doc_text in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i]
                    score = 1 - distance
                    
                    doc = Document(
                        page_content=doc_text,
                        metadata=results['metadatas'][0][i]
                    )
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
    
    def generate_answer(
        self, 
        query: str, 
        context_docs: List[Tuple[Document, float]]
    ) -> Dict:
        """生成答案"""
        # 构建上下文
        context_parts = []
        sources = []
        
        for doc, score in context_docs:
            context_parts.append(f"[相似度：{score:.2f}] {doc.page_content}")
            sources.append({
                "file": doc.metadata.get("source", "未知"),
                "chunk_id": doc.metadata.get("chunk_id", "未知"),
                "score": round(score, 3)
            })
        
        context = "\n\n".join(context_parts)
        
        # 构建提示词
        prompt = self.system_prompt.format(
            context=context,
            question=query
        )
        
        # 调用LLM
        try:
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一位专业的企业知识助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content
            
            # 幻觉检测
            hallucination_check = self._check_hallucination(answer, context)
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": len(context_docs),
                "hallucination_risk": hallucination_check,
                "query": query
            }
            
        except Exception as e:
            error_text = str(e)
            if "401" in error_text or "Incorrect API key" in error_text:
                error_text = (
                    "LLM鉴权失败（401）。请在 .env 中配置正确的 LLM_API_KEY，"
                    "并确认 LLM_API_BASE/LLM_MODEL 与服务端匹配。"
                )

            return {
                "answer": f"生成答案时出错：{error_text}",
                "sources": sources,
                "context_used": len(context_docs),
                "query": query,
                "error": error_text
            }
    
    def _check_hallucination(self, answer: str, context: str) -> Dict:
        """简单的幻觉检测"""
        # 检查是否包含"无法回答"等诚实表述
        honest_phrases = ["无法回答", "没有相关信息", "不清楚", "不知道"]
        is_honest = any(phrase in answer for phrase in honest_phrases)
        
        # 检查是否引用了来源
        has_citation = bool(re.search(r'\[来源[：:]', answer))
        
        return {
            "is_honest": is_honest,
            "has_citation": has_citation,
            "risk_level": "low" if (is_honest or has_citation) else "medium"
        }
    
    def query(self, question: str, with_sources: bool = True) -> Dict:
        """完整查询流程"""
        # 检索
        context_docs = self.retrieve(question)
        
        # 如果没有检索到结果，尝试基于关键词检索
        if not context_docs:
            keywords = self._extract_keywords(question)
            if keywords:
                context_docs = self._keyword_search(keywords, k=config.TOP_K)
        
        if not context_docs:
            return {
                "answer": "未在知识库中找到相关信息，请尝试补充文档或换一种问法。",
                "sources": [],
                "context_used": 0,
                "query": question
            }
        
        # 生成
        result = self.generate_answer(question, context_docs)
        
        if not with_sources:
            result.pop("sources", None)
        
        return result
    
    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "config": {
                "chunk_size": config.CHUNK_SIZE,
                "top_k": config.TOP_K,
                "threshold": config.SIMILARITY_THRESHOLD
            }
        }


# 全局引擎实例
rag_engine = RAGEngine()


if __name__ == "__main__":
    # 测试RAG引擎
    engine = RAGEngine()
    
    # 测试查询
    result = engine.query("公司的年假政策是什么？")
    print(json.dumps(result, ensure_ascii=False, indent=2))