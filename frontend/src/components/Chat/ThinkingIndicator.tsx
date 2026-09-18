import React from 'react';

interface ThinkingIndicatorProps {
  routeHistory?: string[];
  intent?: string;
}

/**
 * Friendly, plain-language "what's happening right now" line shown only while
 * a response is being generated. Replaces a permanent technical flowchart with
 * a single line that disappears once the answer starts arriving — most users
 * want to know the assistant is working, not which pipeline stage it's in.
 */
const STAGE_COPY: { match: string; label: string }[] = [
  { match: 'input_guardrail', label: 'Checking your message...' },
  { match: 'cache_check', label: 'Checking for a quick answer...' },
  { match: 'context_rewriter', label: 'Reading conversation context...' },
  { match: 'query_analyzer', label: 'Understanding your question...' },
  { match: 'web_search', label: 'Searching the web...' },
  { match: 'graph_retriever', label: 'Exploring connected topics...' },
  { match: 'retriever', label: 'Searching your documents...' },
  { match: 'hybrid_multi_engine', label: 'Searching documents and the web...' },
  { match: 'source_fusion', label: 'Gathering the best sources...' },
  { match: 'synthesizer', label: 'Writing your answer...' },
  { match: 'direct_llm', label: 'Writing your answer...' },
  { match: 'critic', label: 'Double-checking the answer...' },
  { match: 'output_guardrail', label: 'Finishing up...' },
];

export const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({ routeHistory = [] }) => {
  const lastStage = routeHistory[routeHistory.length - 1] || '';
  const stage = STAGE_COPY.find((s) => lastStage.toLowerCase().includes(s.match));
  const label = stage?.label || 'Thinking...';

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        fontSize: '13.5px',
        color: 'var(--text-secondary)',
        padding: '2px 0',
      }}
    >
      <span className="thinking-dots">
        <span />
        <span />
        <span />
      </span>
      <span>{label}</span>
    </div>
  );
};
