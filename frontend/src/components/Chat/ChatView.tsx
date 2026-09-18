import React, { useState, useRef, useEffect, Suspense, lazy } from 'react';
import { RefreshCw, CornerDownLeft, AudioLines } from 'lucide-react';
import { ChatMessage, CitationItem, SessionStats } from '../../types';
import { streamChat, getConversation } from '../../services/api';
import { MessageBubble } from './MessageBubble';
import { CitationDrawer } from './CitationDrawer';
import { EmptyState } from './EmptyState';
// Lazily loaded: the orb pulls in three.js (~450kB), which would otherwise sit
// in the main bundle and be downloaded by every user on first paint, including
// the ones who never open voice mode.
const VoiceOverlay = lazy(() =>
  import('../Voice/VoiceOverlay').then((m) => ({ default: m.VoiceOverlay }))
);

interface ChatViewProps {
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  threadId: string | null;
  setThreadId: React.Dispatch<React.SetStateAction<string | null>>;
  stats: SessionStats;
  setStats: React.Dispatch<React.SetStateAction<SessionStats>>;
  onGoToDocuments: () => void;
}

export const ChatView: React.FC<ChatViewProps> = ({
  messages,
  setMessages,
  threadId,
  setThreadId,
  setStats,
  onGoToDocuments,
}) => {
  const [inputQuery, setInputQuery] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeCitation, setActiveCitation] = useState<CitationItem | null>(null);
  const [voiceOpen, setVoiceOpen] = useState(false);

  // Track live streaming progress (feeds the completed message's route_history/intent
  // as a fallback if the "complete" event omits them for some reason)
  const [liveRouteHistory, setLiveRouteHistory] = useState<string[]>([]);
  const [liveIntent, setLiveIntent] = useState<string | undefined>(undefined);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  // Ensure authenticated on mount, then rehydrate history for the saved thread (if any) —
  // the server is the source of truth, so a page refresh needs to re-fetch, not resend.
  useEffect(() => {
    (async () => {
      if (threadId && messages.length === 0) {
        const conversation = await getConversation(threadId);
        if (conversation) {
          setMessages(
            conversation.messages.map((m, idx) => ({
              id: `${conversation.thread_id}-${idx}`,
              role: m.role,
              content: m.content,
              timestamp: m.created_at,
              citations: m.citations || undefined,
            }))
          );
        } else {
          // Thread no longer exists server-side (e.g. different environment) — start fresh.
          setThreadId(null);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSend = async (queryText?: string) => {
    const textToSend = queryText || inputQuery.trim();
    if (!textToSend || isStreaming) return;

    setInputQuery('');
    const userMsgId = `msg-${Date.now()}`;
    const assistantMsgId = `msg-${Date.now() + 1}`;

    const userMessage: ChatMessage = {
      id: userMsgId,
      role: 'user',
      content: textToSend,
      timestamp: new Date().toISOString(),
    };

    const assistantPlaceholder: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isStreaming: true,
      route_history: ['input_guardrail'],
    };

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setIsStreaming(true);
    setLiveRouteHistory(['input_guardrail']);
    setLiveIntent(undefined);

    let streamedText = '';
    let streamedCitations: CitationItem[] = [];

    await streamChat({
      query: textToSend,
      threadId,
      onToken: (token) => {
        streamedText += token;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, content: streamedText }
              : msg
          )
        );
      },
      onRouteUpdate: (data) => {
        if (data.route_history) {
          setLiveRouteHistory(data.route_history);
        }
        if (data.intent) {
          setLiveIntent(data.intent);
        }
        // Also mirror onto the placeholder message itself so MessageBubble can show
        // a friendly "what's happening" line without a separate always-on component.
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  route_history: data.route_history || msg.route_history,
                  intent: (data.intent as any) || msg.intent,
                }
              : msg
          )
        );
      },
      onGuardrailCheck: (data) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  guardrail_status: data.status as any,
                  guardrail_reason: data.reason,
                }
              : msg
          )
        );
      },
      onCitations: (citations) => {
        streamedCitations = citations;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, citations }
              : msg
          )
        );
      },
      onComplete: (data) => {
        setIsStreaming(false);
        if (data.thread_id && data.thread_id !== threadId) {
          setThreadId(data.thread_id);
        }

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  isStreaming: false,
                  content: streamedText || data.answer || msg.content,
                  citations: streamedCitations.length > 0 ? streamedCitations : data.citations,
                  verification_status: data.verification_status,
                  guardrail_status: data.guardrail_status,
                  guardrail_reason: data.guardrail_reason,
                  faithfulness_score: data.faithfulness_score,
                  cache_hit: data.cache_hit,
                  iteration_count: data.iteration_count,
                  route_history: data.route_history || liveRouteHistory,
                  intent: data.intent || (liveIntent as any),
                  raw_query: data.raw_query,
                  rewritten_query: data.rewritten_query,
                  thread_id: data.thread_id,
                }
              : msg
          )
        );

        // Update session stats
        setStats((prev) => ({
          ...prev,
          totalQueries: prev.totalQueries + 1,
          cacheHits: data.cache_hit ? prev.cacheHits + 1 : prev.cacheHits,
          avgFaithfulness:
            data.faithfulness_score !== undefined
              ? (prev.avgFaithfulness * prev.totalQueries + data.faithfulness_score) /
                (prev.totalQueries + 1)
              : prev.avgFaithfulness,
          blockedCount:
            data.guardrail_status === 'BLOCKED'
              ? prev.blockedCount + 1
              : prev.blockedCount,
          piiMaskedCount:
            data.guardrail_status === 'PII_MASKED'
              ? prev.piiMaskedCount + 1
              : prev.piiMaskedCount,
          totalTokensEstimated: prev.totalTokensEstimated + (data.cache_hit ? 0 : 250),
        }));
      },
      onError: (err) => {
        setIsStreaming(false);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? {
                  ...msg,
                  isStreaming: false,
                  content: `⚠️ Error executing request: ${err}`,
                  verification_status: 'INSUFFICIENT_EVIDENCE',
                }
              : msg
          )
        );
      },
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        position: 'relative',
        background: 'var(--bg-base)',
      }}
    >
      {/* Scrollable Message List.
          The inner column is capped at --measure (68ch) and centred. Without the cap
          answer text ran the full window width — ~230 characters per line at 1920px,
          against a comfortable reading measure of 60–75. Long RAG answers are exactly
          where that hurts most. The cap lives on an inner wrapper, not this scroll
          container, so the scrollbar stays at the window edge. */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div
          style={{
            width: '100%',
            maxWidth: 'var(--measure)',
            marginInline: 'auto',
            padding: 'var(--space-6) var(--space-8) var(--space-10)',
            display: 'flex',
            flexDirection: 'column',
            flex: 1,
          }}
        >
        {messages.length === 0 && (
          <EmptyState onAsk={(q) => handleSend(q)} onGoToDocuments={onGoToDocuments} />
        )}

        {/* Message Thread */}
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onCitationClick={(c) => setActiveCitation(c)}
          />
        ))}

        <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Composer — sticky in flow, not absolutely positioned. The old version was
          absolute with a hard-coded 140px scroll reserve underneath it; when the
          composer grew past that the last message hid behind it. */}
      <div
        style={{
          flexShrink: 0,
          borderTop: '1px solid var(--border-subtle)',
          background: 'var(--bg-base)',
          padding: 'var(--space-4) var(--space-8)',
          zIndex: 'var(--z-panel)' as React.CSSProperties['zIndex'],
        }}
      >
        <div style={{ width: '100%', maxWidth: 'var(--measure)', marginInline: 'auto' }}>
        {/* Input Bar */}
        <div
          style={{
            background: 'var(--bg-subtle)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-lg)',
            padding: '8px 12px 8px 16px',
            display: 'flex',
            alignItems: 'flex-end',
            gap: '10px',
            boxShadow: 'var(--shadow-lg)',
          }}
        >
          <textarea
            ref={inputRef}
            rows={1}
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask OmniMind anything (e.g. 'Explain 256-dim embeddings', 'Latest AI news', test prompt injection)..."
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-sans)',
              fontSize: '14.5px',
              lineHeight: 1.5,
              resize: 'none',
              maxHeight: '120px',
              padding: '6px 0',
            }}
          />

          <button
            onClick={() => setVoiceOpen(true)}
            disabled={isStreaming}
            className="btn"
            style={{
              padding: '10px',
              borderRadius: 'var(--radius-md)',
              flexShrink: 0,
              background: 'transparent',
              border: '1px solid var(--border-subtle)',
              color: 'var(--accent-text)',
            }}
            title="Start voice conversation"
            aria-label="Start voice conversation"
          >
            <AudioLines size={16} />
          </button>

          <button
            onClick={() => handleSend()}
            disabled={!inputQuery.trim() || isStreaming}
            className="btn btn-primary"
            style={{
              padding: '10px 14px',
              borderRadius: 'var(--radius-md)',
              flexShrink: 0,
            }}
            title="Send Query (Enter)"
          >
            {isStreaming ? (
              <RefreshCw size={16} className="animate-spin" />
            ) : (
              <>
                <span>Send</span>
                <CornerDownLeft size={14} />
              </>
            )}
          </button>
        </div>
        </div>
      </div>

      {/* Slide-in Citation Drawer */}
      <CitationDrawer
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
      />

      {/* Voice mode. Spoken turns are appended to this same transcript — the
          server persists them into the same thread, so a call and a typed
          conversation are one history rather than two disconnected records. */}
      {voiceOpen && (
        <Suspense fallback={null}>
        <VoiceOverlay
          threadId={threadId}
          onClose={() => setVoiceOpen(false)}
          onTurn={(spokenQuery, spokenAnswer, citations) => {
            const stamp = Date.now();
            setMessages((prev) => [
              ...prev,
              {
                id: `voice-u-${stamp}`,
                role: 'user',
                content: spokenQuery,
                timestamp: new Date().toISOString(),
              },
              {
                id: `voice-a-${stamp}`,
                role: 'assistant',
                content: spokenAnswer,
                timestamp: new Date().toISOString(),
                citations,
                intent: 'voice' as any,
              },
            ]);
          }}
        />
        </Suspense>
      )}
    </div>
  );
};
