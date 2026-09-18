/**
 * OmniMind Frontend Data Models & TypeScript Contracts
 */

export type NavigationTab = 'chat' | 'documents' | 'graph' | 'analytics';

export type GuardrailStatus = 'PASSED' | 'BLOCKED' | 'PII_MASKED';

export type VerificationStatus =
  | 'VERIFIED'
  | 'PARTIALLY_VERIFIED'
  | 'INSUFFICIENT_EVIDENCE'
  | 'CACHED'
  | 'BLOCKED';

export type AgentIntent =
  | 'direct_llm'
  | 'internal_rag'
  | 'web_search'
  | 'hybrid'
  | 'blocked';

export interface CitationItem {
  marker: string;
  chunk_id: string;
  text_snippet: string;
  source_type: 'document' | 'web';
  url?: string | null;
  page?: number | null;
  section?: string | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: CitationItem[];
  verification_status?: VerificationStatus;
  guardrail_status?: GuardrailStatus;
  guardrail_reason?: string | null;
  faithfulness_score?: number;
  route_history?: string[];
  intent?: AgentIntent;
  raw_query?: string | null;
  rewritten_query?: string | null;
  thread_id?: string | null;
  cache_hit?: boolean;
  iteration_count?: number;
  retrieval_trace_id?: string | null;
  isStreaming?: boolean;
}

export interface ChatRequest {
  query: string;
  thread_id?: string | null;
  max_iterations?: number;
}

export interface ConversationSummary {
  thread_id: string;
  title: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationDetail {
  thread_id: string;
  title: string;
  summary: string | null;
  messages: Array<{
    role: 'user' | 'assistant';
    content: string;
    citations?: CitationItem[] | null;
    created_at: string;
  }>;
}

export interface ChatResponse {
  answer: string;
  citations: CitationItem[];
  verification_status: VerificationStatus;
  guardrail_status: GuardrailStatus;
  guardrail_reason?: string | null;
  faithfulness_score: number;
  route_history: string[];
  raw_query?: string | null;
  rewritten_query?: string | null;
  thread_id?: string | null;
  cache_hit: boolean;
  iteration_count: number;
  intent: AgentIntent;
  retrieval_trace_id?: string | null;
}

export interface DocumentItem {
  id: string;
  filename: string;
  status: 'UPLOADED' | 'PARSED' | 'CHUNKED' | 'EMBEDDED' | 'FAILED';
  s3_key?: string;
  created_at?: string;
  chunk_count?: number;
  file_size?: number;
}

export interface GraphNode {
  id: string;
  name: string;
  type: 'CONCEPT' | 'TECHNOLOGY' | 'ORGANIZATION' | 'DOCUMENT' | 'ENTITY';
  summary?: string;
  frequency?: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

export interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  relation: string;
  weight?: number;
}

export interface KnowledgeGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface SearchResultItem {
  chunk_id: string;
  text: string;
  dense_score?: number | null;
  sparse_score?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  source_stage: string;
  metadata?: Record<string, any>;
}

export interface SearchResponse {
  query: string;
  total_results: number;
  trace_id: string;
  results: SearchResultItem[];
}

export interface SystemHealth {
  status: string;
  postgres: boolean;
  redis: boolean;
  minio: boolean;
  model: string;
  embedding_model: string;
}

export interface SessionStats {
  totalQueries: number;
  cacheHits: number;
  avgFaithfulness: number;
  blockedCount: number;
  piiMaskedCount: number;
  intentCounts: Record<string, number>;
  totalTokensEstimated: number;
}
