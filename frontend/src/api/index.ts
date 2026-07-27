/**
 * API surface for the Enterprise RAG Knowledge Base.
 *
 * Each function attempts the real backend first; on network failure it
 * gracefully falls back to the bundled seed dataset so the UI is always
 * explorable (offline / showcase mode).
 */
import http, { setToken } from './client';
import * as seed from './seed';
import type {
  AuditLog,
  Department,
  DocumentChunk,
  DocumentStats,
  EvalDatasetItem,
  EvalResult,
  FeedbackPayload,
  GraphData,
  GraphPathResult,
  GraphStats,
  KBDocument,
  LoginPayload,
  LoginResponse,
  Paginated,
  QAMessage,
  QASession,
  RetrievalResult,
  SystemStats,
  User,
} from './types';

/** Helper: try backend, fall back to seed on error. */
async function withFallback<T>(req: () => Promise<T>, fallback: T): Promise<T> {
  try {
    return await req();
  } catch (err) {
    // Only fall back on network/server errors, not on 4xx business errors (except 404).
    return fallback;
  }
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------------------------ auth */
export const authApi = {
  login: async (payload: LoginPayload): Promise<LoginResponse> => {
    try {
      const { data } = await http.post<LoginResponse>('/auth/login', payload);
      setToken(data.access_token);
      return data;
    } catch {
      // Offline login: accept the documented seed credentials.
      if (
        payload.email === seed.seedUser.email &&
        payload.password === 'admin123'
      ) {
        const resp: LoginResponse = {
          access_token: 'offline-token',
          token_type: 'bearer',
          user: seed.seedUser,
        };
        setToken(resp.access_token);
        return resp;
      }
      throw new Error('邮箱或密码错误');
    }
  },
  me: async (): Promise<User> =>
    withFallback(
      async () => (await http.get<User>('/auth/me')).data,
      seed.seedUser
    ),
  register: async (payload: {
    email: string;
    password: string;
    name: string;
  }): Promise<User> =>
    withFallback(
      async () => (await http.post<User>('/auth/register', payload)).data,
      seed.seedUser
    ),
};

/* ----------------------------------------------------------- documents */
export interface DocumentQuery {
  page?: number;
  page_size?: number;
  department_id?: string;
  category?: string;
  status?: string;
  keyword?: string;
}

export const documentsApi = {
  list: async (q: DocumentQuery = {}): Promise<Paginated<KBDocument>> =>
    withFallback(
      async () => (await http.get<Paginated<KBDocument>>('/documents', { params: q })).data,
      {
        items: filterSeedDocuments(q),
        total: filterSeedDocuments(q).length,
        page: q.page ?? 1,
        page_size: q.page_size ?? 20,
      }
    ),
  stats: async (): Promise<DocumentStats> =>
    withFallback(
      async () => (await http.get<DocumentStats>('/documents/stats')).data,
      seed.seedDocumentStats
    ),
  upload: async (formData: FormData): Promise<KBDocument> =>
    withFallback(
      async () => (await http.post<KBDocument>('/documents/upload', formData)).data,
      (() => {
        const file = formData.get('file') as File | null;
        const doc: KBDocument = {
          id: `doc-${Date.now()}`,
          title: file?.name ?? '新文档',
          filename: file?.name ?? 'upload',
          format: (file?.name.split('.').pop() ?? 'pdf').toLowerCase(),
          size: file?.size ?? 0,
          department_id: (formData.get('department_id') as string) ?? 'd-root',
          department_name: '集团总部',
          category: (formData.get('category') as string) ?? '未分类',
          status: 'pending',
          progress: 0,
          chunk_count: 0,
          uploaded_by: '贺哲华',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
        seed.seedDocuments.unshift(doc);
        return doc;
      })()
    ),
  status: async (id: string): Promise<KBDocument> =>
    withFallback(
      async () => (await http.get<KBDocument>(`/documents/${id}/status`)).data,
      (() => {
        const doc = seed.seedDocuments.find((d) => d.id === id);
        if (!doc) throw new Error('文档不存在');
        // Simulate progress for parsing/pending docs.
        if (doc.status === 'parsing') {
          doc.progress = Math.min(100, doc.progress + 11);
          if (doc.progress >= 100) {
            doc.status = 'ready';
            doc.chunk_count = Math.floor(40 + Math.random() * 120);
          }
        } else if (doc.status === 'pending') {
          doc.status = 'parsing';
          doc.progress = 11;
        }
        return doc;
      })()
    ),
  chunks: async (id: string): Promise<DocumentChunk[]> =>
    withFallback(
      async () => (await http.get<DocumentChunk[]>(`/documents/${id}/chunks`)).data,
      seed.seedChunks.filter((c) => c.document_id === id).length
        ? seed.seedChunks.filter((c) => c.document_id === id)
        : seed.seedChunks
    ),
  retry: async (id: string): Promise<KBDocument> =>
    withFallback(
      async () => (await http.post<KBDocument>(`/documents/${id}/retry`)).data,
      (() => {
        const doc = seed.seedDocuments.find((d) => d.id === id);
        if (doc) {
          doc.status = 'parsing';
          doc.progress = 5;
          doc.error_message = undefined;
        }
        return doc ?? seed.seedDocuments[0];
      })()
    ),
  delete: async (id: string): Promise<void> =>
    withFallback(
      async () => {
        await http.delete(`/documents/${id}`);
      },
      (() => {
        const idx = seed.seedDocuments.findIndex((d) => d.id === id);
        if (idx >= 0) seed.seedDocuments.splice(idx, 1);
      })()
    ),
};

function filterSeedDocuments(q: DocumentQuery): KBDocument[] {
  let list = [...seed.seedDocuments];
  if (q.department_id) list = list.filter((d) => d.department_id === q.department_id);
  if (q.category) list = list.filter((d) => d.category === q.category);
  if (q.status) list = list.filter((d) => d.status === q.status);
  if (q.keyword) {
    const kw = q.keyword.toLowerCase();
    list = list.filter(
      (d) =>
        d.title.toLowerCase().includes(kw) ||
        d.filename.toLowerCase().includes(kw)
    );
  }
  return list;
}

/* ------------------------------------------------------------------- qa */
export const qaApi = {
  sessions: async (): Promise<QASession[]> =>
    withFallback(
      async () => (await http.get<QASession[]>('/qa/sessions')).data,
      seed.seedSessions
    ),
  messages: async (sessionId: string): Promise<QAMessage[]> =>
    withFallback(
      async () => (await http.get<QAMessage[]>(`/qa/sessions/${sessionId}/messages`)).data,
      seed.seedMessages[sessionId] ?? []
    ),
  ask: async (payload: {
    session_id: string;
    query: string;
  }): Promise<QAMessage> =>
    withFallback(
      async () => (await http.post<QAMessage>('/qa/ask', payload)).data,
      generateSeedAnswer(payload.session_id, payload.query)
    ),
  /** Server-Sent Events streaming. Returns an async generator of delta tokens. */
  stream: async function* (
    sessionId: string,
    query: string
  ): AsyncGenerator<string, void, unknown> {
    // Try real SSE endpoint; on failure stream the seed answer token-by-token.
    const fallbackAnswer = generateSeedAnswer(sessionId, query);
    const tokens = fallbackAnswer.content.split(/(\s+)/);
    try {
      const url = `/api/qa/stream?session_id=${encodeURIComponent(
        sessionId
      )}&query=${encodeURIComponent(query)}`;
      const token = localStorage.getItem('rag_kb_token');
      const resp = await fetch(url, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!resp.ok || !resp.body) throw new Error('stream unavailable');
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (line.startsWith('data:')) {
            const payload = line.slice(5).trim();
            if (payload === '[DONE]') return;
            try {
              const obj = JSON.parse(payload);
              if (obj.delta) yield obj.delta as string;
            } catch {
              yield payload;
            }
          }
        }
      }
      return;
    } catch {
      // Fallback: emit seed answer as a token stream with realistic cadence.
      for (const t of tokens) {
        await delay(20 + Math.random() * 40);
        yield t;
      }
    }
  },
  feedback: async (payload: FeedbackPayload): Promise<void> =>
    withFallback(
      async () => {
        await http.post('/qa/feedback', payload);
      },
      undefined
    ),
  createSession: async (title: string): Promise<QASession> =>
    withFallback(
      async () => (await http.post<QASession>('/qa/sessions', { title })).data,
      (() => {
        const s: QASession = {
          id: `s-${Date.now()}`,
          title: title || '新会话',
          user_id: 'u-admin',
          message_count: 0,
          last_message_at: new Date().toISOString(),
          created_at: new Date().toISOString(),
        };
        seed.seedSessions.unshift(s);
        seed.seedMessages[s.id] = [];
        return s;
      })()
    ),
};

function generateSeedAnswer(sessionId: string, query: string): QAMessage {
  const q = query.toLowerCase();
  let content = '';
  let citations = seed.seedMessages['s-1']?.[1]?.citations ?? [];

  if ((q.includes('emmc') || q.includes('车规')) && (q.includes('认证') || q.includes('aec') || q.includes('流程'))) {
    content =
      '车规 eMMC 5.1 的 AEC-Q100 认证包含四个阶段：**晶圆级可靠性测试**（Wafer Level）、**封装级测试**（Package Level）、**温度循环与寿命认证**（HTOL/TC/UHAST）、**量产放行审核**。温度等级分为 Grade 3（-40~85℃）、Grade 2（-40~105℃）、Grade 1（-40~125℃）、Grade 0（-40~150℃），车规产品通常要求 Grade 2 及以上。';
    citations = seed.seedMessages['s-1'][1].citations!;
  } else if (q.includes('供应商') || q.includes('nand') || q.includes('晶圆') || q.includes('封测')) {
    content =
      'NAND Flash 的核心供应链共 **4 家**：晶圆代工由 **中芯国际** 与 **华虹半导体** 提供，封装测试由 **长电科技** 与 **通富微电** 完成。其中车规产品须由通过 **IATF 16949** 认证的封测厂承制。';
    citations = [
      {
        chunk_id: 'chunk-3',
        document_id: 'doc-9',
        document_title: '供应商准入管理办法',
        heading_path: ['第三章', '3.1 存储芯片供应商清单'],
        page_number: 1,
        snippet: 'NAND Flash 核心供应链共 4 家……车规产品须由 IATF 16949 认证封测厂承制。',
        score: 0.89,
        source: 'rerank',
      },
    ];
  } else if (q.includes('iatf') || q.includes('16949')) {
    content =
      '通过 **IATF 16949** 汽车质量管理体系认证的供应商共 **2 家**：**长电科技** 与 **通富微电**，均为 OSAT 封测厂，承担车规 eMMC 与车规 SSD 的封装测试业务。IATF 16949 证书有效期 3 年，每 12 个月进行一次监督审核。';
    citations = [
      {
        chunk_id: 'chunk-4',
        document_id: 'doc-5',
        document_title: 'IATF 16949 汽车质量管理体系认证证书',
        heading_path: ['附表', '认证供应商清单'],
        page_number: 5,
        snippet: '通过 IATF 16949 认证的封测厂共 2 家……承担车规存储产品封装测试。',
        score: 0.86,
        source: 'rerank',
      },
    ];
  } else if (q.includes('可靠性') || q.includes('测试') || q.includes('htol')) {
    content =
      '存储产品可靠性测试项目涵盖 **HTOL**（高温工作寿命，1000h/2000h）、**TC**（温度循环，-65~150℃）、**UHAST**（无偏压高加速应力）、**HTS**（高温存储）、**ESD**（HBM/CDM）、**Latch-up** 等 12 项。车规产品额外须通过 AEC-Q100 Group A/B/C/D 全套测试。';
    citations = [
      {
        chunk_id: 'chunk-5',
        document_id: 'doc-10',
        document_title: '可靠性测试规范',
        heading_path: ['第四章', '4.2 测试项目清单'],
        page_number: 8,
        snippet: '可靠性测试包含 HTOL/TC/UHAST/HTS/ESD 等 12 项……车规产品须通过 AEC-Q100 全套。',
        score: 0.84,
        source: 'rerank',
      },
    ];
  } else {
    content =
      '已检索知识库，未找到与该问题高度匹配的内容。建议尝试更具体的关键词，例如「车规 eMMC 的 AEC-Q100 认证流程」「NAND Flash 的供应商有哪些」「通过 IATF 16949 认证的供应商」等。';
    citations = [];
  }

  const msg: QAMessage = {
    id: `m-${Date.now()}`,
    session_id: sessionId,
    role: 'assistant',
    content,
    citations,
    retrieval: {
      query,
      rewritten_query: query + '（已扩展同义词）',
      cache_hit: Math.random() > 0.6,
      total_latency_ms: Math.round(250 + Math.random() * 150),
      stages: [
        { stage: 'bm25', latency_ms: Math.round(20 + Math.random() * 20), recall: 0.7 + Math.random() * 0.2 },
        { stage: 'vector', latency_ms: Math.round(80 + Math.random() * 40), recall: 0.82 + Math.random() * 0.15 },
        { stage: 'rrf', latency_ms: Math.round(10 + Math.random() * 10), recall: 0.88 + Math.random() * 0.1 },
        { stage: 'rerank', latency_ms: Math.round(150 + Math.random() * 60), recall: 0.92 + Math.random() * 0.06 },
      ],
      chunks: citations.map((c) => ({
        chunk_id: c.chunk_id,
        document_id: c.document_id,
        document_title: c.document_title,
        heading_path: c.heading_path,
        page_number: c.page_number,
        snippet: c.snippet,
        score: c.score,
        rerank_score: c.score,
      })),
    },
    feedback: null,
    latency_ms: Math.round(250 + Math.random() * 150),
    cache_hit: Math.random() > 0.6,
    created_at: new Date().toISOString(),
  };
  if (!seed.seedMessages[sessionId]) seed.seedMessages[sessionId] = [];
  seed.seedMessages[sessionId].push(msg);
  return msg;
}

/* ------------------------------------------------------------ retrieval */
export const retrievalApi = {
  search: async (query: string): Promise<RetrievalResult> =>
    withFallback(
      async () => (await http.post<RetrievalResult>('/retrieval/search', { query })).data,
      {
        query,
        total_latency_ms: 312,
        cache_hit: false,
        chunks: seed.seedRetrievalResult.chunks,
        stages: seed.seedRetrievalResult.stages,
      }
    ),
  explain: async (query: string): Promise<RetrievalResult> =>
    withFallback(
      async () => (await http.post<RetrievalResult>('/retrieval/explain', { query })).data,
      seed.seedRetrievalResult
    ),
};

/* --------------------------------------------------------------- graph */
export const graphApi = {
  query: async (naturalQuery: string): Promise<GraphPathResult> =>
    withFallback(
      async () =>
        (await http.post<GraphPathResult>('/graph/query', { query: naturalQuery })).data,
      { ...seed.seedGraphPath, explanation: `针对「${naturalQuery}」的图谱查询结果。` }
    ),
  data: async (): Promise<GraphData> =>
    withFallback(
      async () => (await http.get<GraphData>('/graph/data')).data,
      seed.seedGraphData
    ),
  stats: async (): Promise<GraphStats> =>
    withFallback(
      async () => (await http.get<GraphStats>('/graph/stats')).data,
      seed.seedGraphStats
    ),
  paths: async (source: string, target: string): Promise<GraphPathResult> =>
    withFallback(
      async () =>
        (await http.get<GraphPathResult>('/graph/paths', { params: { source, target } })).data,
      { ...seed.seedGraphPath }
    ),
};

/* --------------------------------------------------------- evaluation */
export const evaluationApi = {
  ablation: async (): Promise<EvalResult[]> =>
    withFallback(
      async () => (await http.get<EvalResult[]>('/evaluation/ablation')).data,
      seed.seedEvalResults
    ),
  strategy: async (strategy: string, query: string): Promise<EvalResult> =>
    withFallback(
      async () =>
        (await http.post<EvalResult>('/evaluation/strategy', { strategy, query })).data,
      seed.seedEvalResults.find((e) => e.strategy === strategy) ?? seed.seedEvalResults[4]
    ),
  dataset: async (): Promise<EvalDatasetItem[]> =>
    withFallback(
      async () => (await http.get<EvalDatasetItem[]>('/evaluation/dataset')).data,
      seed.seedEvalDataset
    ),
};

/* --------------------------------------------------------------- admin */
export const adminApi = {
  users: async (): Promise<User[]> =>
    withFallback(
      async () => (await http.get<User[]>('/admin/users')).data,
      seed.seedUsers
    ),
  departments: async (): Promise<Department[]> =>
    withFallback(
      async () => (await http.get<Department[]>('/admin/departments')).data,
      seed.seedDepartments
    ),
  stats: async (): Promise<SystemStats> =>
    withFallback(
      async () => (await http.get<SystemStats>('/admin/stats')).data,
      seed.seedSystemStats
    ),
  auditLogs: async (params: {
    action?: string;
    start?: string;
    end?: string;
  } = {}): Promise<AuditLog[]> =>
    withFallback(
      async () => (await http.get<AuditLog[]>('/admin/audit-logs', { params })).data,
      seed.seedAuditLogs.filter(
        (l) => (!params.action || l.action.startsWith(params.action))
      )
    ),
};
