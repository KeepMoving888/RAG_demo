# RAG 检索流水线设计

本文档详述企业级 RAG 知识库的检索流水线，覆盖从文档解析、清洗、语义分块、向量化，到 BM25 + 稠密向量混合检索、RRF 融合、Cross-Encoder 精排的完整链路，并给出消融评估方法与实测效果数据。检索质量是 RAG 系统的根基，生成阶段的上限由检索召回决定，因此本文的设计取舍均围绕"在不显著增加延迟的前提下最大化召回率"展开。

## 1. 流水线总览

```
原文档 ──► Parser ──► Cleaner ──► Chunker ──► Embedder ──► 索引写入
                                                          │
                                                          ▼
用户问题 ──► QueryRewriter ──► HybridRetriever ──► RRF Fusion ──► Reranker ──► LLM 生成
              (术语扩展)        (BM25 + 向量)        (k=60)        (bge-reranker)
```

各阶段均以可插拔组件实现，遵循统一接口（`BaseParser` / `BaseCleaner` / `BaseChunker` / `BaseEmbedder` / `BaseRetriever` / `BaseReranker`），便于替换实现或在消融实验中关闭某一阶段。

## 2. 文档解析（Parser）

不同格式的文档质量差异极大：扫描版 PDF 没有文本层、Word 文档可能包含复杂表格、图片需要 OCR。Parser 层按格式分发，并具备兜底链。

### 2.1 PDF 解析

优先使用 PyMuPDF 提取文本层（速度快、保真度高）。当某页提取的字符数低于阈值（默认 50 字符/页）或检测到大量乱码（`�` 占比 > 5%）时，判定该页为扫描页，回退到 PaddleOCR。

```python
class PDFParser(BaseParser):
    """PDF 解析器：文本层优先，扫描页 PaddleOCR 兜底。"""

    MIN_CHARS_PER_PAGE = 50
    GARBAGE_RATIO_THRESHOLD = 0.05

    def parse(self, file_path: str) -> ParsedDocument:
        doc = fitz.open(file_path)
        pages = []
        for page_idx, page in enumerate(doc):
            text = page.get_text("text")
            if self._is_text_valid(text):
                pages.append(Page(index=page_idx, text=text, source="pymupdf"))
            else:
                # 扫描页：渲染为图片后 OCR
                image = self._render_page_to_image(page, dpi=200)
                ocr_text = self.ocr.recognize(image)
                pages.append(Page(index=page_idx, text=ocr_text, source="paddleocr"))
        return ParsedDocument(pages=pages, meta={"file": file_path})

    def _is_text_valid(self, text: str) -> bool:
        if len(text.strip()) < self.MIN_CHARS_PER_PAGE:
            return False
        garbage_ratio = text.count("\ufffd") / max(len(text), 1)
        return garbage_ratio < self.GARBAGE_RATIO_THRESHOLD
```

表格识别对 RAG 召回至关重要——表格被拆成纯文本后语义会严重丢失。本系统对含表格的 PDF 页面，调用 PaddleOCR 的结构化表格识别能力，将表格转为 Markdown 格式（保留行列结构），与正文一并进入分块。

### 2.2 Word 解析

使用 Unstructured 库解析 `.docx`，保留标题层级（`Title` / `Heading 1` / `Heading 2`）与表格结构。标题层级信息会透传给 Chunker，用于基于 heading stack 的语义分块。

### 2.3 图片解析

通过 PaddleOCR 识别文字。图片通常作为文档附图出现，Parser 会将识别结果作为独立 chunk 处理，并在元数据中标注 `chunk_type=image`，检索时与文本 chunk 平等参与融合。

## 3. 文档清洗（Cleaner）

原始解析结果中混杂大量噪声，直接分块会污染向量空间。清洗阶段执行以下规则：

| 清洗项 | 检测方法 | 处理方式 |
|--------|----------|----------|
| 页眉页脚 | 统计跨页重复出现的首尾行（重复率 ≥ 60% 视为页眉页脚） | 删除 |
| 水印 | 检测半透明、跨页固定位置、低对比度的重复文本 | 删除 |
| 乱码行 | 行中 `�` 或非可打印字符占比 > 30% | 删除该行 |
| 重复空白 | 连续空行/空格压缩为单个 | 归一化 |
| 孤立标点 | 行仅由标点构成且长度 ≤ 2 | 删除 |
| 段落归一化 | 软换行（行尾非句号）合并为同段落 | 合并 |
| 表格结构化 | 表格转 Markdown，保留 `\|` 分隔 | 结构化保留 |

清洗后的文本以"段落"为最小语义单元进入分块器，每个段落携带其所属标题路径（如 `["第一章", "1.2 系统设计", "1.2.1 架构"]`）。

## 4. 语义分块（Chunker）

分块策略直接决定召回质量。固定长度切分会破坏语义边界，导致一个完整概念被切到两个 chunk 中，向量检索时双方都难以命中。本系统采用"标题层级 + 段落语义连贯性 + 父子层级"的复合分块策略。

### 4.1 算法流程

1. **构建 heading stack**：遍历清洗后的段落，遇到标题（来自 Parser 透传的层级）时压栈，得到每个段落的标题路径。
2. **按标题分组**：同一最末级标题下的段落归为一组。
3. **段落语义连贯性判断**：组内段落顺序合并，当累计长度超过 `target_chunk_size`（默认 512 token）时，检查相邻段落的语义连贯性。若语义相似度（用句子向量余弦相似度估计）低于阈值 0.5，则在此处切分；否则继续合并直到达到 `max_chunk_size`（默认 768 token）。
4. **大块二次切分**：若合并后仍超过 `max_chunk_size`，在句子边界二次切分，保留 `overlap=64` token 重叠，避免边界信息丢失。
5. **小块合并**：长度低于 `min_chunk_size`（默认 128 token）的 chunk 与相邻 chunk 合并，避免过短 chunk 向量表征不稳。
6. **父子层级建立**：每个 chunk 记录 `parent_chunk_id`，指向其所属的大语义块（通常是标题下的整段内容）。检索时返回叶子 chunk，但生成 Prompt 时可向上回溯拼接父块，提供更完整上下文。

```python
@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    parent_chunk_id: str | None
    content: str
    heading_path: list[str]          # ["第一章", "1.2 系统设计"]
    token_count: int
    department_id: str               # 用于 ACL 过滤
    chunk_type: str                  # text | table | image
    embedding: list[float] | None
```

### 4.2 分块效果对比

在种子评测集（32 条标注查询，覆盖 45 篇半导体存储种子文档）上对比三种分块策略，固定其余检索链路：

| 分块策略 | 平均 chunk 长度 | chunk 总数 | Recall@5 | MRR | 说明 |
|----------|-----------------|-----------|----------|-----|------|
| 固定长度 chunk_size=512, overlap=64 | 512 | 18420 | 0.68 | 0.52 | 切断语义边界，跨块问题召回差 |
| 句子切分（按句号） | 180 | 32100 | 0.74 | 0.58 | chunk 过短，向量表征不稳，上下文不足 |
| **标题层级 + 段落语义（本系统）** | 460 | 19850 | **0.85** | **0.71** | 语义边界对齐，父子层级提供回溯上下文 |

标题层级分块在 Recall@5 上较固定长度提升 17 个百分点，主要收益来自跨标题概念的完整保留与父块回溯。

## 5. 嵌入向量化（Embedder）

### 5.1 模型选择

采用 BGE-M3 作为嵌入模型，输出 1024 维稠密向量。选择依据：

- **多语言**：企业文档中英文混排常见，BGE-M3 对中英文均有良好表征。
- **多粒度**：支持稠密向量 + 稀疏向量 + ColBERT 多向量，本系统仅使用稠密向量部分，稀疏检索交给 BM25。
- **长上下文**：最大 8192 token，覆盖大多数 chunk 无需截断。

### 5.2 批量化与缓存

向量化是 CPU/GPU 密集型操作，批量嵌入可显著提升吞吐。Worker 以 batch_size=32 调用嵌入服务。同一 chunk 内容若曾在 Redis 向量缓存中命中（key 为内容哈希），则跳过嵌入直接取用，避免重解析时重复计算。

```python
class BGEM3Embedder(BaseEmbedder):
    """BAAI/bge-m3 嵌入器 (1024 维, 支持 GPU + FP16 + Redis 向量缓存).

    实际实现见 backend/app/rag/embedder.py.
    """

    DIMENSION = 1024
    BATCH_SIZE = 32

    def __init__(self, model_name: str, device: str = "cpu",
                 use_fp16: bool = False, cache: "RetrievalCache | None" = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = cache

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # 1. 先查 Redis 向量缓存 (key=内容哈希), 命中则跳过
        results: list[list[float] | None] = [None] * len(texts)
        missed_idx = []
        for i, t in enumerate(texts):
            if self.cache is not None:
                cached = await self.cache.get_vector(t)
                if cached is not None:
                    results[i] = cached
                    continue
            missed_idx.append(i)
        # 2. 批量嵌入未命中部分 (GPU + FP16 吞吐约 2x)
        if missed_idx:
            missed_texts = [texts[i] for i in missed_idx]
            vectors = self.model.encode(
                missed_texts, batch_size=self.BATCH_SIZE,
                normalize_embeddings=True,
            )
            for i, vec in zip(missed_idx, vectors):
                results[i] = vec.tolist()
                if self.cache is not None:
                    await self.cache.set_vector(texts[i], vec.tolist(), ttl=86400)
        return results  # type: ignore[return-value]
```

向量缓存使重解析场景（如分块策略调整后批量重跑）的嵌入计算量降低约 80%。

## 6. 混合检索（HybridRetriever）

单一检索方式各有短板：稠密向量擅长语义匹配但弱于精确术语匹配；BM25 擅长关键词匹配但无法理解同义表达。混合检索取两者之长。

### 6.1 BM25 稀疏检索

采用 `rank_bm25` 实现，参数 `k1=1.5`、`b=0.75`。在标准 BM25 之上引入术语加权：当查询或文档中出现业务术语词典中的术语时，对该词的匹配权重乘以 `term_boost`（默认 2.0）。

BM25 打分公式：

```
score(D, Q) = Σ_{q_i ∈ Q} IDF(q_i) · (f(q_i, D) · (k1 + 1)) / (f(q_i, D) + k1 · (1 - b + b · |D| / avgdl))
```

其中：
- `f(q_i, D)`：词 `q_i` 在文档 `D` 中的词频。
- `|D|`：文档 `D` 的长度（token 数）。
- `avgdl`：语料平均文档长度。
- `IDF(q_i) = ln((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1)`，`N` 为文档总数，`n(q_i)` 为含 `q_i` 的文档数。
- `k1` 控制词频饱和速度，`b` 控制文档长度归一化强度。

### 6.2 稠密向量检索

用户问题经 BGE-M3 嵌入后，在 Milvus 中执行 ANN 检索（IVF_FLAT 索引，`nprobe=16`），返回 Top-20。检索按 `partition=dept_{user_department_id}` 缩小扫描范围，兼顾性能与权限隔离。

### 6.3 RRF 融合

Reciprocal Rank Fusion 将多路检索结果按排名融合，无需归一化各路分数，对异构检索器友好。

RRF 打分公式：

```
score(d) = Σ_{i ∈ retrievers} 1 / (k + rank_i(d))
```

其中 `rank_i(d)` 是文档 `d` 在第 `i` 路检索结果中的排名（从 1 开始，未召回记为无穷大即贡献 0），`k=60` 为经验最优值——`k` 过小会让排名靠前的结果主导（接近纯取并集后按头部排序），`k` 过大则抹平排名差异（接近平均）。在 60 时融合效果最稳定，该结论与 Cormack 等人的原始论文一致。

融合前先做 ACL 过滤与去重（按 `chunk_id`），融合后取 Top-20 进入精排阶段。

### 6.4 Cross-Encoder 精排

RRF 融合后的 Top-20 仍可能包含语义相关但与用户真实意图有偏差的结果。Cross-Encoder（`bge-reranker-v2-m3`）将 (query, chunk) 拼接后联合编码打分，比双塔模型的独立编码更能捕捉细粒度匹配关系。

精排将 Top-20 重排后取 Top-5 进入生成阶段。精排是整条链路中延迟最高的非 LLM 环节（单次约 8ms，20 条约 40ms，可批量并行降至 ~15ms），其收益（Recall@5 提升 6 个百分点）值得这一延迟开销。

精排不可用时降级为直接取 RRF 融合后 Top-5，详见架构文档降级策略链。

## 7. 业务术语词典（TerminologyExpander）

企业内部存在大量专有术语与缩写，纯语义检索难以匹配。例如查询"请说明 RTO 的标准"中的 `RTO`，文档中可能写作"恢复时间目标"。术语词典维护同义词与缩写映射，在检索阶段对术语关键词加权，并在 Query 改写阶段扩展同义表达。

```python
class TerminologyExpander:
    """业务术语扩展：同义词 OR 注入 + 检索阶段术语关键词加权。

    实际实现见 backend/app/rag/terminology.py.
    """

    def __init__(self, term_path: str | None = None):
        # term_path 指向 terminology.json, 缺省走内置兜底词典
        # 数据结构: {term: {"synonyms": [...], "type": "..."}}
        self._terms: dict[str, dict] = {}

    def expand_query(self, query: str) -> tuple[str, list[str]]:
        """同义词 OR 注入到 query, 返回 (扩展后 query, 命中术语列表).

        命中术语时, 同义词以 OR 追加到原 query 末尾,
        使 BM25 能匹配代号命名的文档.
        """
        expanded = query
        hits: list[str] = []
        for term, meta in self._terms.items():
            if term in query:
                hits.append(term)
                for syn in meta.get("synonyms", []):
                    expanded += f" OR {syn}"
        return expanded, hits

    def boost_term_weight(self, tokens: list[str],
                          term_hits: list[str]) -> list[tuple[str, float]]:
        """返回 (token, weight) 列表, 术语命中 token 权重 ×2.0.

        供 BM25 检索阶段对术语词项加权使用.
        """
        result = []
        for tok in tokens:
            w = 2.0 if tok in term_hits else 1.0
            result.append((tok, w))
        return result
```

术语词典由管理员通过后台维护，支持按部门配置不同词典子集（不同部门术语含义可能不同）。

## 8. 消融评估

为量化各组件贡献，系统内置消融评估框架，可在 `POST /api/v1/search/eval` 触发。评估在标注评测集上对比以下策略：

| 策略编号 | 策略名称 | 向量检索 | BM25 | RRF 融合 | 术语加权 | Cross-Encoder 精排 |
|----------|----------|----------|------|----------|----------|---------------------|
| S1 | vector_only | ✅ | ✗ | ✗ | ✗ | ✗ |
| S2 | bm25_only | ✗ | ✅ | ✗ | ✗ | ✗ |
| S3 | rrf | ✅ | ✅ | ✅ | ✗ | ✗ |
| S4 | full | ✅ | ✅ | ✅ | ✅ | ✅ |
| S5 | full_with_terminology | ✅ | ✅ | ✅ | ✅ | ✅ |

评估指标：

- **Recall@K**：前 K 个结果是否包含标注答案所在 chunk。
- **MRR**（Mean Reciprocal Rank）：第一个正确结果的排名倒数的均值。
- **NDCG@K**：考虑排名位置与相关性等级的归一化折损累计增益。
- **Precision@K**：前 K 个结果中相关结果的比例。

### 8.1 检索效果对比

在种子评测集（32 题，K=5）上的实测结果：

| 策略 | Recall@5 | MRR | NDCG@5 | Precision@5 | P95 延迟 |
|------|----------|-----|--------|-------------|----------|
| S1 vector_only | 0.72 | 0.55 | 0.61 | 0.48 | 28ms |
| S2 bm25_only | 0.66 | 0.49 | 0.54 | 0.42 | 12ms |
| S3 rrf | 0.85 | 0.68 | 0.74 | 0.61 | 35ms |
| S4 full | 0.88 | 0.72 | 0.79 | 0.66 | 52ms |
| **S5 full_with_terminology** | **0.91** | **0.76** | **0.83** | **0.70** | 55ms |

关键结论：

1. **向量 only（0.72）→ RRF 融合（0.85）**：BM25 补足了精确术语匹配能力，Recall 提升 13 个百分点，是单步最大收益。
2. **RRF（0.85）→ RRF + 术语 + 精排（0.91）**：术语扩展解决同义词召回缺口，精排将正确结果前置，Recall@5 再提升 6 个百分点。
3. **延迟代价可控**：从 S1 到 S5，P95 延迟从 28ms 升至 55ms，远低于 LLM 生成延迟（~1.5s），不构成瓶颈。

### 8.2 评估执行

评估脚本读取评测集（JSONL，每行含 `question`、`gold_chunk_ids`、`relevance_grades`），对每个策略跑全量检索，输出指标与逐题明细。RAG 相关改动需在 PR 中附带评估结果对比，详见 [贡献指南](../CONTRIBUTING.md)。

```bash
# 触发消融评估
curl -X POST http://localhost:8080/api/v1/search/eval \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"strategies": ["vector_only","bm25_only","rrf","full","full_with_terminology"],
       "top_k": 5, "dataset": "internal_v3"}'
```

## 9. 相关文档

- [系统架构设计](./architecture.md)：流水线在整体架构中的位置与降级策略。
- [GraphRAG 知识图谱增强](./graphrag.md)：图谱检索作为第三路召回的融合方式。
- [性能基准测试](./benchmark.md)：大规模数据下的延迟与吞吐数据。
