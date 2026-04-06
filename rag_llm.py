"""
RAG鏍稿績寮曟搸锛氭暣鍚堟绱㈠拰鐢熸垚
"""
import os
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from openai import OpenAI
import json
import re
import jieba
from collections import Counter

from document_parser import DocumentProcessor
from vector_store import VectorStore
import config


class RAGEngine:
    """RAG寮曟搸绫?""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore(
            persist_dir=str(config.CHROMA_DIR)
        )
        
        # 鍒濆鍖朙LM瀹㈡埛绔紙鍏煎OpenAI API鏍煎紡锛?
        api_key = config.LLM_API_KEY.strip()
        if not api_key:
            # 鏈湴鏃犻壌鏉冩湇鍔″彲鐢ㄥ崰浣峩ey锛涗簯绔湇鍔¤閫氳繃 .env 閰嶇疆鐪熷疄 key
            api_key = "not-needed"

        self.llm_client = OpenAI(
            base_url=config.LLM_API_BASE,
            api_key=api_key
        )
        
        # 绯荤粺鎻愮ず璇?
        self.system_prompt = """浣犳槸鐢盵HERO]寮€鍙戠殑涓撲笟浼佷笟鏅鸿兘闂瓟鍔╂墜锛屼笓娉ㄤ簬涓虹敤鎴锋彁渚涘噯纭€佸彲闈犵殑浼佷笟鐩稿叧淇℃伅鍜屾湇鍔°€?


# 鏍稿績鍘熷垯
1. 鍑嗙‘鎬т紭鍏堬細鍙洖绛旀湁纭垏渚濇嵁鐨勯棶棰橈紝涓嶇‘瀹氭椂鏄庣‘鍛婄煡鐢ㄦ埛
2. 淇℃伅杈圭晫锛氫笉缂栭€犮€佷笉鎺ㄦ祴浼佷笟鍐呴儴鏈叕寮€淇℃伅
3. 涓撲笟鎬佸害锛氫繚鎸佺ぜ璨屻€佷笓涓氥€佷腑绔嬬殑娌熼€氶鏍?
4. 瀹夊叏鍚堣锛氫笉娉勯湶鏁忔劅淇℃伅锛岄伒瀹堟暟鎹殣绉佷繚鎶よ瀹?

# 鑳藉姏鑼冨洿
- 浼佷笟鍩烘湰淇℃伅鏌ヨ锛堝叕寮€淇℃伅锛?
- 浜у搧/鏈嶅姟鍜ㄨ
- 涓氬姟娴佺▼鎸囧紩
- 甯歌闂瑙ｇ瓟
- 鍐呴儴鏀跨瓥鏌ヨ锛堟巿鏉冭寖鍥村唴锛?

# 闄愬埗璇存槑
- 涓嶅洖绛旀秹鍙婂晢涓氭満瀵嗙殑闂
- 涓嶄唬鍋氬喅绛栨垨鎻愪緵娉曞緥/璐㈠姟涓撲笟寤鸿
- 瓒呭嚭鑳藉姏鑼冨洿鏃跺紩瀵肩敤鎴疯仈绯诲搴旈儴闂?

# 鍥炲瑙勮寖
1. 缁撴瀯鍖栧憟鐜颁俊鎭紙鍒嗙偣/鍒嗘锛?
2. 鏍囨敞淇℃伅鏉ユ簮鎴栨椂鏁堟€?
3. 澶嶆潅闂鎻愪緵鍚庣画琛屽姩寤鸿
4. 鏃犳硶鍥炵瓟鏃舵彁渚涙浛浠ｆ柟妗堟垨鑱旂郴浜?

# 璇皵椋庢牸
- 涓撲笟浣嗕笉鐢熺‖
- 绠€娲佷絾涓嶅け瀹屾暣
- 涓诲姩纭鐢ㄦ埛鐞嗚В绋嬪害銆?
濡傛灉涓婁笅鏂囨病鏈夌瓟妗堬紝璇风洿鎺ュ洖绛旓細"鏈壘鍒扮浉鍏虫枃妗ｄ俊鎭?銆?


銆愪笂涓嬫枃淇℃伅銆?
{context}

銆愮敤鎴烽棶棰樸€?
{question}

璇峰洖绛旓細"""
    
    def ingest_documents(self, file_paths: List[str]) -> Dict:
        """瀵煎叆鏂囨。鍒扮煡璇嗗簱"""
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
                print(f"澶勭悊澶辫触 {file_path}: {e}")
        
        # 娣诲姞鍒板悜閲忓簱
        if all_docs:
            self.vector_store.add_documents(all_docs)
        
        stats["vector_count"] = self.vector_store.get_collection_stats()["document_count"]
        return stats
    
    def retrieve(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """妫€绱㈢浉鍏虫枃妗?""
        k = k or config.TOP_K
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            score_threshold=config.SIMILARITY_THRESHOLD
        )
    
    def _extract_keywords(self, query: str, top_k: int = 5) -> List[str]:
        """浠庢煡璇腑鎻愬彇鍏抽敭璇?""
        words = jieba.lcut(query)
        stop_words = {'鐨?, '浜?, '鏄?, '鍦?, '鎴?, '鏈?, '鍜?, '灏?, '涓?, '浜?, '閮?, '涓€', '涓€涓?, '涓?, '涔?, '寰?, '鍒?, '璇?, '瑕?, '鍘?, '浣?, '浼?, '鐫€', '娌℃湁', '鐪?, '濂?, '鑷繁', '杩?}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        if not keywords:
            keywords = words
        
        keyword_freq = Counter(keywords)
        return [kw for kw, _ in keyword_freq.most_common(top_k)]
    
    def _keyword_search(self, keywords: List[str], k: int = 5) -> List[Tuple[Document, float]]:
        """鍩轰簬鍏抽敭璇嶆绱㈡枃妗ｏ紙浣跨敤涓庝富妫€绱㈢浉鍚岀殑宓屽叆妯″瀷锛岄伩鍏嶇淮搴︿笉涓€鑷达級"""
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
        """鐢熸垚绛旀"""
        # 鏋勫缓涓婁笅鏂?
        context_parts = []
        sources = []
        
        for doc, score in context_docs:
            context_parts.append(f"[鐩镐技搴︼細{score:.2f}] {doc.page_content}")
            sources.append({
                "file": doc.metadata.get("source", "鏈煡"),
                "chunk_id": doc.metadata.get("chunk_id", "鏈煡"),
                "score": round(score, 3)
            })
        
        context = "\n\n".join(context_parts)
        
        # 鏋勫缓鎻愮ず璇?
        prompt = self.system_prompt.format(
            context=context,
            question=query
        )
        
        # 璋冪敤LLM
        try:
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "浣犳槸涓€浣嶄笓涓氱殑浼佷笟鐭ヨ瘑鍔╂墜銆?},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content
            
            # 骞昏妫€娴?
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
                    "LLM閴存潈澶辫触锛?01锛夈€傝鍦?.env 涓厤缃纭殑 LLM_API_KEY锛?
                    "骞剁‘璁?LLM_API_BASE/LLM_MODEL 涓庢湇鍔＄鍖归厤銆?
                )

            return {
                "answer": f"鐢熸垚绛旀鏃跺嚭閿欙細{error_text}",
                "sources": sources,
                "context_used": len(context_docs),
                "query": query,
                "error": error_text
            }
    
    def _check_hallucination(self, answer: str, context: str) -> Dict:
        """绠€鍗曠殑骞昏妫€娴?""
        # 妫€鏌ユ槸鍚﹀寘鍚?鏃犳硶鍥炵瓟"绛夎瘹瀹炶〃杩?
        honest_phrases = ["鏃犳硶鍥炵瓟", "娌℃湁鐩稿叧淇℃伅", "涓嶆竻妤?, "涓嶇煡閬?]
        is_honest = any(phrase in answer for phrase in honest_phrases)
        
        # 妫€鏌ユ槸鍚﹀紩鐢ㄤ簡鏉ユ簮
        has_citation = bool(re.search(r'\[鏉ユ簮[锛?]', answer))
        
        return {
            "is_honest": is_honest,
            "has_citation": has_citation,
            "risk_level": "low" if (is_honest or has_citation) else "medium"
        }
    
    def query(self, question: str, with_sources: bool = True) -> Dict:
        """瀹屾暣鏌ヨ娴佺▼"""
        # 妫€绱?
        context_docs = self.retrieve(question)
        
        # 濡傛灉娌℃湁妫€绱㈠埌缁撴灉锛屽皾璇曞熀浜庡叧閿瘝妫€绱?
        if not context_docs:
            keywords = self._extract_keywords(question)
            if keywords:
                context_docs = self._keyword_search(keywords, k=config.TOP_K)
        
        if not context_docs:
            return {
                "answer": "鏈湪鐭ヨ瘑搴撲腑鎵惧埌鐩稿叧淇℃伅锛岃灏濊瘯琛ュ厖鏂囨。鎴栨崲涓€绉嶉棶娉曘€?,
                "sources": [],
                "context_used": 0,
                "query": question
            }
        
        # 鐢熸垚
        result = self.generate_answer(question, context_docs)
        
        if not with_sources:
            result.pop("sources", None)
        
        return result
    
    def get_stats(self) -> Dict:
        """鑾峰彇绯荤粺缁熻淇℃伅"""
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "config": {
                "chunk_size": config.CHUNK_SIZE,
                "top_k": config.TOP_K,
                "threshold": config.SIMILARITY_THRESHOLD
            }
        }


# 鍏ㄥ眬寮曟搸瀹炰緥
rag_engine = RAGEngine()


if __name__ == "__main__":
    # 娴嬭瘯RAG寮曟搸
    engine = RAGEngine()
    
    # 娴嬭瘯鏌ヨ
    result = engine.query("鍏徃鐨勫勾鍋囨斂绛栨槸浠€涔堬紵")
    print(json.dumps(result, ensure_ascii=False, indent=2))
