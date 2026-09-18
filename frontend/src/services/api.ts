import {
  ChatResponse,
  CitationItem,
  ConversationDetail,
  ConversationSummary,
  DocumentItem,
  KnowledgeGraphData,
  SearchResponse,
  SystemHealth,
} from '../types';

const API_BASE = '';

const TOKEN_KEY = 'omnimind_token';
const TOKEN_EXPIRY_KEY = 'omnimind_token_expires_at';

/** Notified whenever the session ends, so the app can return to the sign-in screen
 *  from anywhere — including a 401 raised deep inside a background fetch. */
type SessionEndedHandler = () => void;
let onSessionEnded: SessionEndedHandler | null = null;

export function setSessionEndedHandler(handler: SessionEndedHandler | null): void {
  onSessionEnded = handler;
}

export function getAuthToken(): string | null {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) return null;

  // Treat an expired token as absent. Without this the app keeps sending a token
  // it knows is dead and discovers the problem as a failed action mid-task.
  const expiresAt = Number(localStorage.getItem(TOKEN_EXPIRY_KEY) || 0);
  if (expiresAt && Date.now() >= expiresAt) {
    removeAuthToken();
    return null;
  }
  return token;
}

export function setAuthToken(token: string, expiresInSeconds?: number): void {
  localStorage.setItem(TOKEN_KEY, token);
  if (expiresInSeconds && expiresInSeconds > 0) {
    // 30s of slack so a request started just before expiry isn't rejected in flight.
    localStorage.setItem(
      TOKEN_EXPIRY_KEY,
      String(Date.now() + (expiresInSeconds - 30) * 1000)
    );
  } else {
    localStorage.removeItem(TOKEN_EXPIRY_KEY);
  }
}

export function removeAuthToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(TOKEN_EXPIRY_KEY);
}

/** Central handler for "the server says this session is over". */
export function handleUnauthorized(): void {
  removeAuthToken();
  onSessionEnded?.();
}

/** Pulls the human-readable message out of a FastAPI error body.
 *  `detail` is a string for HTTPException and an array for a 422 validation
 *  error, and showing the raw JSON of either is how users end up reading
 *  `{"detail":"Not authenticated"}` on screen. */
export async function readApiError(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0];
      return typeof first?.msg === 'string'
        ? first.msg.replace(/^Value error,\s*/, '')
        : fallback;
    }
    return fallback;
  } catch {
    return fallback;
  }
}

/**
 * Common fetch headers
 */
function getHeaders(isJson: boolean = true): HeadersInit {
  const headers: Record<string, string> = {};
  if (isJson) {
    headers['Content-Type'] = 'application/json';
  }
  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

/**
 * Health check
 */
export async function getSystemHealth(): Promise<SystemHealth> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    const data = await res.json();
    const services = data.services || {};
    // The endpoint reports "ok"; the UI compares against "HEALTHY". Without this
    // mapping the header sat on "Connecting" forever even with every service up.
    const healthy = String(data.status).toLowerCase() === 'ok';
    return {
      status: healthy ? 'HEALTHY' : 'DEGRADED',
      postgres: services.postgres === 'up',
      redis: services.redis === 'up',
      minio: services.minio === 'up',
      model: 'gemini-3.7-flash',
      embedding_model: 'gemini-embedding-2 (256-dim)',
    };
  } catch {
    return {
      status: 'OFFLINE',
      postgres: false,
      redis: false,
      minio: false,
      model: 'gemini-3.7-flash',
      embedding_model: 'gemini-embedding-2 (256-dim)',
    };
  }
}

/**
 * Authentication: Quick signup/login helper
 */
export interface AuthUser {
  id: string;
  email: string;
  is_active: boolean;
  created_at: string;
}

/** Thrown for an auth failure the user can act on — the message is display-ready.
 *  `status` lets the UI offer a specific recovery (e.g. 409 → switch to sign-in). */
export class AuthError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

async function submitCredentials(
  path: '/auth/login' | '/auth/signup',
  email: string,
  password: string,
  fallbackMessage: string
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
    });
  } catch {
    throw new AuthError("Can't reach the server. Check that the API is running.");
  }

  if (!res.ok) {
    throw new AuthError(await readApiError(res, fallbackMessage), res.status);
  }

  const data = await res.json();
  setAuthToken(data.access_token, data.expires_in);
}

export async function login(email: string, password: string): Promise<void> {
  return submitCredentials('/auth/login', email, password, 'Incorrect email or password.');
}

export async function signup(email: string, password: string): Promise<void> {
  return submitCredentials('/auth/signup', email, password, 'Could not create that account.');
}

export function logout(): void {
  removeAuthToken();
}

/** Resolves the signed-in user, or null when there is no valid session.
 *  Used on boot to decide between the app and the sign-in screen. */
export async function getCurrentUser(): Promise<AuthUser | null> {
  if (!getAuthToken()) return null;
  try {
    const res = await fetch(`${API_BASE}/auth/me`, { headers: getHeaders() });
    if (res.status === 401 || res.status === 403) {
      removeAuthToken();
      return null;
    }
    if (!res.ok) return null;
    return (await res.json()) as AuthUser;
  } catch {
    // A network blip is not a signed-out user — keep the token and let the
    // caller retry rather than dumping them back to the login screen.
    return null;
  }
}

/**
 * Real-time SSE Chat Streaming client for POST /chat/stream
 */
export async function streamChat({
  query,
  threadId = null,
  maxIterations = 2,
  onToken,
  onRouteUpdate,
  onGuardrailCheck,
  onCitations,
  onComplete,
  onError,
}: {
  query: string;
  threadId?: string | null;
  maxIterations?: number;
  onToken: (token: string) => void;
  onRouteUpdate: (data: { status: string; intent?: string; route_history?: string[]; message?: string }) => void;
  onGuardrailCheck: (data: { status: string; reason?: string | null }) => void;
  onCitations: (citations: CitationItem[]) => void;
  onComplete: (data: Partial<ChatResponse>) => void;
  onError: (error: string) => void;
}): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: getHeaders(true),
      body: JSON.stringify({
        query,
        thread_id: threadId,
        max_iterations: maxIterations,
      }),
    });

    if (res.status === 401 || res.status === 403) {
      handleUnauthorized();
      throw new Error('Your session has expired. Please sign in again.');
    }

    if (!res.ok) {
      // Never surface the raw body — that is how `{"detail":"Not authenticated"}`
      // ended up rendered as an assistant message.
      throw new Error(await readApiError(res, `Server error (${res.status})`));
    }

    if (!res.body) {
      throw new Error('ReadableStream not supported by response');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      let currentEvent = 'message';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        if (trimmed.startsWith('event:')) {
          currentEvent = trimmed.replace('event:', '').trim();
        } else if (trimmed.startsWith('data:')) {
          const dataStr = trimmed.replace('data:', '').trim();
          try {
            const parsed = JSON.parse(dataStr);
            if (currentEvent === 'token') {
              onToken(parsed.text || '');
            } else if (currentEvent === 'route_update') {
              onRouteUpdate(parsed);
            } else if (currentEvent === 'guardrail_check') {
              onGuardrailCheck(parsed);
            } else if (currentEvent === 'citations') {
              onCitations(parsed.citations || []);
            } else if (currentEvent === 'complete') {
              onComplete(parsed);
            } else if (currentEvent === 'error') {
              onError(parsed.error || 'Unknown streaming error');
            }
          } catch {
            // Raw text fallback
            if (currentEvent === 'token') {
              onToken(dataStr);
            }
          }
        }
      }
    }
  } catch (err: any) {
    onError(err.message || 'Failed to connect to agent stream');
  }
}

/**
 * Synchronous / fallback Chat Endpoint POST /chat
 */
export async function sendChat(
  query: string,
  threadId: string | null = null,
  maxIterations = 2
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: getHeaders(true),
    body: JSON.stringify({
      query,
      thread_id: threadId,
      max_iterations: maxIterations,
    }),
  });

  if (res.status === 401 || res.status === 403) {
    handleUnauthorized();
    throw new Error('Your session has expired. Please sign in again.');
  }

  if (!res.ok) {
    throw new Error(await readApiError(res, `Chat failed (${res.status})`));
  }

  return res.json();
}

/**
 * Conversations API — lets the frontend rehydrate history after a page refresh,
 * since the server (not the client) is the source of truth for chat history.
 */
export async function listConversations(): Promise<ConversationSummary[]> {
  const res = await fetch(`${API_BASE}/conversations`, {
    headers: getHeaders(true),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function getConversation(threadId: string): Promise<ConversationDetail | null> {
  const res = await fetch(`${API_BASE}/conversations/${threadId}`, {
    headers: getHeaders(true),
  });
  if (!res.ok) return null;
  return res.json();
}

/**
 * Documents API
 */
export async function listDocuments(): Promise<DocumentItem[]> {
  const res = await fetch(`${API_BASE}/documents`, {
    headers: getHeaders(true),
  });
  if (!res.ok) {
    // Return sample demo documents if auth/backend isn't seeded yet
    return [
      {
        id: 'doc-001',
        filename: 'OmniMind_Architecture_v2.pdf',
        status: 'EMBEDDED',
        chunk_count: 24,
        created_at: new Date(Date.now() - 3600000).toISOString(),
      },
      {
        id: 'doc-002',
        filename: 'Matryoshka_Gemini_Embedding2.pdf',
        status: 'EMBEDDED',
        chunk_count: 18,
        created_at: new Date(Date.now() - 7200000).toISOString(),
      },
      {
        id: 'doc-003',
        filename: 'GraphRAG_MultiHop_Benchmark.md',
        status: 'EMBEDDED',
        chunk_count: 12,
        created_at: new Date(Date.now() - 14400000).toISOString(),
      },
    ];
  }
  return res.json();
}

export async function uploadDocument(file: File): Promise<DocumentItem> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/documents`, {
    method: 'POST',
    headers: getHeaders(false), // don't set Content-Type for FormData
    body: formData,
  });

  if (res.status === 401 || res.status === 403) {
    handleUnauthorized();
    throw new Error('Your session has expired. Please sign in again.');
  }
  if (!res.ok) {
    throw new Error(await readApiError(res, `Upload failed (${res.status})`));
  }

  return res.json();
}

export async function getDocument(documentId: string): Promise<DocumentItem> {
  const res = await fetch(`${API_BASE}/documents/${documentId}`, {
    headers: getHeaders(true),
  });
  if (res.status === 401 || res.status === 403) {
    handleUnauthorized();
    throw new Error('Your session has expired. Please sign in again.');
  }
  if (!res.ok) {
    throw new Error(await readApiError(res, `Could not check document status (${res.status})`));
  }
  return res.json();
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${documentId}`, {
    method: 'DELETE',
    headers: getHeaders(true),
  });

  if (res.status === 401 || res.status === 403) {
    handleUnauthorized();
    throw new Error('Your session has expired. Please sign in again.');
  }
  if (!res.ok && res.status !== 204) {
    throw new Error(await readApiError(res, `Delete failed (${res.status})`));
  }
}

/**
 * Voice API — the catalogue comes from the server so it can't drift from the
 * allowlist that actually validates the choice on connect.
 */
export interface VoiceOption {
  name: string;
  character: string;
}

export async function listVoices(): Promise<{ default: string; voices: VoiceOption[] }> {
  try {
    const res = await fetch(`${API_BASE}/voice/voices`, { headers: getHeaders(true) });
    if (!res.ok) throw new Error('unavailable');
    return await res.json();
  } catch {
    // The picker is a convenience; a failed fetch should not block the call.
    return { default: 'Puck', voices: [] };
  }
}

/**
 * Knowledge Graph API
 */
export async function getKnowledgeGraph(): Promise<KnowledgeGraphData> {
  const res = await fetch(`${API_BASE}/graph`, {
    headers: getHeaders(true),
  });
  if (!res.ok) {
    // Fallback seed graph for demo visualization
    return {
      nodes: [
        { id: 'OmniMind', name: 'OmniMind', type: 'TECHNOLOGY', summary: 'Adaptive Multi-Source Enterprise Agent' },
        { id: 'FastAPI', name: 'FastAPI', type: 'TECHNOLOGY', summary: 'High-performance Async REST API' },
        { id: 'LangGraph', name: 'LangGraph', type: 'TECHNOLOGY', summary: 'StateGraph Multi-Agent Orchestrator' },
        { id: 'Jina CLIP v2', name: 'Jina CLIP v2', type: 'TECHNOLOGY', summary: '256-dim Matryoshka Multimodal Embeddings' },
        { id: 'Pinecone', name: 'Pinecone', type: 'TECHNOLOGY', summary: 'Vector Index for Dense Chunk Storage' },
        { id: 'Tavily AI', name: 'Tavily AI', type: 'TECHNOLOGY', summary: 'Clean Markdown Web Search' },
        { id: 'Cohere Rerank', name: 'Cohere Rerank', type: 'TECHNOLOGY', summary: 'Cross-encoder reranker (v3.5)' },
        { id: 'GraphRAG', name: 'GraphRAG', type: 'CONCEPT', summary: 'Knowledge Graph Multi-Hop Discovery' },
        { id: 'RAGAS', name: 'RAGAS', type: 'CONCEPT', summary: 'Automated Evaluation Suite' },
        { id: 'Guardrails', name: 'Native Guardrails', type: 'CONCEPT', summary: 'Zero-token Injection and PII Blocker' },
      ],
      edges: [
        { source: 'OmniMind', target: 'FastAPI', relation: 'BUILT_WITH' },
        { source: 'OmniMind', target: 'LangGraph', relation: 'ORCHESTRATED_BY' },
        { source: 'OmniMind', target: 'Jina CLIP v2', relation: 'EMBEDDED_WITH' },
        { source: 'OmniMind', target: 'Pinecone', relation: 'INDEXED_IN' },
        { source: 'OmniMind', target: 'Tavily AI', relation: 'AUGMENTED_BY' },
        { source: 'OmniMind', target: 'Cohere Rerank', relation: 'RERANKED_WITH' },
        { source: 'OmniMind', target: 'GraphRAG', relation: 'USES_DISCOVERY' },
        { source: 'OmniMind', target: 'RAGAS', relation: 'EVALUATED_BY' },
        { source: 'OmniMind', target: 'Guardrails', relation: 'PROTECTED_BY' },
      ],
    };
  }
  return res.json();
}

export async function searchGraph(q: string): Promise<any[]> {
  const res = await fetch(`${API_BASE}/graph/search?q=${encodeURIComponent(q)}`, {
    headers: getHeaders(true),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function getNeighborhood(entity: string, maxHops = 2): Promise<any[]> {
  const res = await fetch(`${API_BASE}/graph/neighborhood?entity=${encodeURIComponent(entity)}&max_hops=${maxHops}`, {
    headers: getHeaders(true),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function findPaths(source: string, target: string): Promise<string[][]> {
  const res = await fetch(`${API_BASE}/graph/paths?source=${encodeURIComponent(source)}&target=${encodeURIComponent(target)}`, {
    headers: getHeaders(true),
  });
  if (!res.ok) return [];
  return res.json();
}

/**
 * Hybrid Search API POST /search
 */
export async function searchHybrid(query: string, topK = 5): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: getHeaders(true),
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) {
    throw new Error('Search failed');
  }
  return res.json();
}
