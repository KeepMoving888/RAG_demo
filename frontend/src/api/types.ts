/** Domain type definitions for the Enterprise RAG Knowledge Base frontend. */

export type Role = 'admin' | 'editor' | 'viewer';

export interface User {
  id: string;
  email: string;
  name: string;
  role: Role;
  department_id: string | null;
  department_name?: string;
  avatar_url?: string;
  is_active: boolean;
  created_at: string;
  last_login_at?: string;
}

export interface Department {
  id: string;
  name: string;
  parent_id: string | null;
  code: string;
  member_count: number;
  children?: Department[];
}

export type DocumentStatus =
  | 'pending'
  | 'parsing'
  | 'ready'
  | 'failed';

export interface KBDocument {
  id: string;
  title: string;
  filename: string;
  format: string;
  size: number;
  department_id: string;
  department_name: string;
  category: string;
  status: DocumentStatus;
  progress: number;
  chunk_count: number;
  error_message?: string;
  uploaded_by: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  chunk_index: number;
  heading_path: string[];
  page_number: number;
  content: string;
  token_count: number;
  bm25_terms?: string[];
}

export interface DocumentStats {
  total: number;
  parsing: number;
  ready: number;
  failed: number;
  total_chunks: number;
  total_size: number;
}

export type MessageRole = 'user' | 'assistant';

export interface Citation {
  chunk_id: string;
  document_id: string;
  document_title: string;
  heading_path: string[];
  page_number: number;
  snippet: string;
  score: number;
  source?: 'bm25' | 'vector' | 'rrf' | 'rerank';
}

export interface QAMessage {
  id: string;
  session_id: string;
  role: MessageRole;
  content: string;
  citations?: Citation[];
  retrieval?: RetrievalExplain;
  feedback?: 'up' | 'down' | null;
  latency_ms?: number;
  cache_hit?: boolean;
  created_at: string;
}

export interface QASession {
  id: string;
  title: string;
  user_id: string;
  message_count: number;
  last_message_at: string;
  created_at: string;
}

export interface RetrievalStage {
  stage: 'bm25' | 'vector' | 'rrf' | 'rerank';
  latency_ms: number;
  recall: number;
}

export interface RetrievalExplain {
  query: string;
  rewritten_query: string;
  cache_hit: boolean;
  total_latency_ms: number;
  stages: RetrievalStage[];
  chunks: RetrievedChunk[];
}

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  document_title: string;
  heading_path: string[];
  page_number: number;
  snippet: string;
  score: number;
  bm25_score?: number;
  vector_score?: number;
  rrf_score?: number;
  rerank_score?: number;
}

export interface RetrievalResult {
  query: string;
  total_latency_ms: number;
  cache_hit: boolean;
  chunks: RetrievedChunk[];
  stages: RetrievalStage[];
}

export type EntityType =
  | 'Product'
  | 'Department'
  | 'Person'
  | 'Policy'
  | 'Supplier'
  | 'Certification'
  | 'Patent';

export interface GraphNode {
  id: string;
  label: string;
  type: EntityType;
  properties: Record<string, string | number>;
  source_chunks: number;
}

export type RelationType =
  | 'BELONGS_TO'
  | 'MANUFACTURES'
  | 'CERTIFIED_BY'
  | 'SUPPLIES'
  | 'AUTHORED_BY'
  | 'REFERENCES'
  | 'GOVERNED_BY'
  | 'AUDITED_BY'
  | 'PARTICIPATES_IN'
  | 'INVENTED_BY';

export interface GraphRelation {
  id: string;
  source: string;
  target: string;
  type: RelationType;
  weight: number;
  properties?: Record<string, string | number>;
}

export interface GraphData {
  nodes: GraphNode[];
  links: { source: string; target: string; type: RelationType; weight: number }[];
}

export interface GraphStats {
  node_count: number;
  relation_count: number;
  type_distribution: Record<EntityType, number>;
  relation_distribution: Record<RelationType, number>;
}

export interface GraphPathResult {
  cypher: string;
  records: Record<string, unknown>[];
  explanation: string;
}

export interface EvalMetrics {
  recall_at_5: number;
  mrr: number;
  ndcg_at_5: number;
  precision_at_5: number;
}

export interface EvalResult {
  strategy: string;
  description: string;
  metrics: EvalMetrics;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
}

export interface LatencyPoint {
  strategy: string;
  p50: number;
  p95: number;
  p99: number;
}

export interface EvalDatasetItem {
  id: string;
  query: string;
  relevant_chunk_ids: string[];
  expected_answer: string;
  difficulty: 'easy' | 'medium' | 'hard';
}

export interface SystemStats {
  user_count: number;
  document_count: number;
  session_count: number;
  retrieval_count: number;
  cache_hit_rate: number;
  graph_node_count: number;
  parse_failed_count: number;
  rate_limited_count: number;
  trend: { date: string; value: number }[];
}

export interface AuditLog {
  id: string;
  user_id: string;
  user_name: string;
  action: string;
  resource_type: string;
  resource_id: string;
  detail: string;
  ip: string;
  created_at: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Paginated<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface FeedbackPayload {
  message_id: string;
  feedback: 'up' | 'down';
  comment?: string;
}
