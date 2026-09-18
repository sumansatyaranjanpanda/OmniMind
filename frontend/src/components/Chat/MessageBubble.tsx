import React from 'react';
import { AlertTriangle, Copy, Check } from 'lucide-react';
import { ChatMessage, CitationItem } from '../../types';
import { AnswerMarkdown } from './AnswerMarkdown';
import { GuardrailBadge } from './GuardrailBadge';
import { ReasoningTrace } from './ReasoningTrace';
import { ThinkingIndicator } from './ThinkingIndicator';

interface MessageBubbleProps {
  message: ChatMessage;
  onCitationClick: (citation: CitationItem) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onCitationClick,
}) => {
  const isUser = message.role === 'user';
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className="animate-fade-in-up"
      style={{
        display: 'flex',
        // The user's question is a bubble; the answer is the page. An answer can run
        // several hundred words, and a balloon around that much prose fights reading.
        maxWidth: isUser ? '80%' : '100%',
        width: isUser ? 'auto' : '100%',
        alignSelf: isUser ? 'flex-end' : 'stretch',
        marginBottom: 'var(--space-6)',
      }}
    >
      {/* Bubble Container */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-2)',
          width: '100%',
        }}
      >
        {/* Main Card */}
        <div
          style={{
            background: isUser ? 'var(--bg-muted)' : 'transparent',
            border: 'none',
            borderRadius: isUser ? 'var(--radius-lg)' : '0',
            padding: isUser ? 'var(--space-3) var(--space-4)' : '0',
            color: 'var(--text-primary)',
            fontSize: 'var(--text-md)',
            lineHeight: 'var(--leading-body)',
            position: 'relative',
          }}
        >
          {/* Guardrail notice — only shown when something actually happened (masked/blocked),
              not on every ordinary passed message. */}
          {!isUser && message.guardrail_status && message.guardrail_status !== 'PASSED' && (
            <div style={{ marginBottom: '10px' }}>
              <GuardrailBadge status={message.guardrail_status} reason={message.guardrail_reason} />
            </div>
          )}

          {message.isStreaming && !message.content ? (
            <ThinkingIndicator routeHistory={message.route_history} />
          ) : isUser ? (
            // The user's own message is plain text. Rendering it as markdown would let a
            // question containing a `#` or `*` come back visually reformatted.
            <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {message.content}
            </div>
          ) : (
            <div style={{ wordBreak: 'break-word' }}>
              <AnswerMarkdown
                content={message.content}
                citations={message.citations}
                onCitationClick={onCitationClick}
              />
              {message.isStreaming && <span className="typing-cursor" />}
            </div>
          )}

          {/* Citation Pills at bottom of bubble */}
          {!isUser && message.citations && message.citations.length > 0 && (
            <div
              style={{
                marginTop: '14px',
                paddingTop: '10px',
                borderTop: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                flexWrap: 'wrap',
              }}
            >
              <span style={{ fontSize: '11.5px', color: 'var(--text-muted)', fontWeight: 600 }}>
                Sources:
              </span>
              {message.citations.map((c, i) => (
                <button
                  key={i}
                  className="citation-ref"
                  onClick={() => onCitationClick(c)}
                  style={{ fontSize: '11.5px', padding: '2px 8px' }}
                >
                  {c.marker || `[^${i + 1}]`} {c.section ? `• ${c.section}` : ''}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Trust warning — surfaced only when it matters, not on every message */}
        {!isUser && !message.isStreaming && message.verification_status === 'INSUFFICIENT_EVIDENCE' && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              paddingLeft: '4px',
              fontSize: '11.5px',
              color: 'var(--warning)',
            }}
          >
            <AlertTriangle size={13} />
            <span>This answer may not be fully backed by your sources.</span>
          </div>
        )}

        {/* Footer: timestamp, copy, and the optional reasoning trace */}
        {!isUser && !message.isStreaming && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingLeft: '4px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
            <button
              onClick={handleCopy}
              className="btn btn-ghost"
              style={{ padding: '3px 6px', fontSize: '11px', gap: '4px' }}
              title="Copy answer"
            >
              {copied ? <Check size={12} color="var(--success)" /> : <Copy size={12} />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
          </div>
        )}

        {!isUser && !message.isStreaming && (
          <ReasoningTrace
            routeHistory={message.route_history}
            intent={message.intent}
            cacheHit={message.cache_hit}
            faithfulnessScore={message.faithfulness_score}
            rewrittenQuery={message.rewritten_query}
            iterationCount={message.iteration_count}
          />
        )}
      </div>

    </div>
  );
};
