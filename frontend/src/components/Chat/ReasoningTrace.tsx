import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Sparkles } from 'lucide-react';

interface ReasoningTraceProps {
  routeHistory?: string[];
  intent?: string;
  cacheHit?: boolean;
  faithfulnessScore?: number;
  rewrittenQuery?: string | null;
  iterationCount?: number;
}

const STEP_LABELS: Record<string, string> = {
  input_guardrail: 'Checked your message',
  cache_check: 'Checked for a cached answer',
  context_rewriter: 'Read conversation context',
  query_analyzer: 'Understood your question',
  retriever: 'Searched your documents',
  graph_retriever: 'Explored connected topics',
  web_search: 'Searched the web',
  hybrid_multi_engine: 'Searched documents and the web',
  source_fusion: 'Gathered the best sources',
  synthesizer: 'Wrote the answer',
  direct_llm: 'Wrote the answer',
  critic: 'Double-checked the answer',
  query_rewriter: 'Refined the search',
  output_guardrail: 'Finished up',
};

const INTENT_LABELS: Record<string, string> = {
  direct_llm: 'Answered directly, no lookup needed',
  internal_rag: 'Answered using your documents',
  web_search: 'Answered using a web search',
  hybrid: 'Answered using documents and the web',
  blocked: 'Blocked by safety checks',
};

/**
 * Collapsed by default — the technical trace (route steps, confidence score,
 * cache hit, contextual rewrite) lives here instead of permanently on screen.
 * Most people never open this; it's here for the person who wants to know
 * exactly how an answer was produced.
 */
export const ReasoningTrace: React.FC<ReasoningTraceProps> = ({
  routeHistory = [],
  intent,
  cacheHit = false,
  faithfulnessScore,
  rewrittenQuery,
  iterationCount,
}) => {
  const [open, setOpen] = useState(false);

  if (routeHistory.length === 0) return null;

  const steps = routeHistory
    .map((r) => STEP_LABELS[r] || null)
    .filter((label, idx, arr): label is string => !!label && arr.indexOf(label) === idx);

  return (
    <div style={{ marginTop: '10px' }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          background: 'transparent',
          border: 'none',
          color: 'var(--text-muted)',
          fontSize: '11.5px',
          cursor: 'pointer',
          padding: '2px 0',
        }}
      >
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        <span>{open ? 'Hide reasoning' : 'Show reasoning'}</span>
      </button>

      {open && (
        <div
          className="animate-fade-in-up"
          style={{
            marginTop: '8px',
            padding: '12px 14px',
            background: 'var(--bg-deep)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            fontSize: '12px',
            color: 'var(--text-secondary)',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
          }}
        >
          {intent && (
            <div>{INTENT_LABELS[intent] || intent}</div>
          )}

          {steps.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {steps.map((label, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ color: 'var(--success)' }}>✓</span>
                  <span>{label}</span>
                </div>
              ))}
            </div>
          )}

          {rewrittenQuery && (
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '6px' }}>
              <Sparkles size={13} color="var(--accent)" style={{ marginTop: '1px', flexShrink: 0 }} />
              <span>Understood your question as: <strong>"{rewrittenQuery}"</strong></span>
            </div>
          )}

          <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', paddingTop: '4px', borderTop: '1px solid var(--border-subtle)' }}>
            {cacheHit && <span>⚡ Instant answer (previously answered)</span>}
            {faithfulnessScore !== undefined && faithfulnessScore > 0 && (
              <span>Answer confidence: {(faithfulnessScore * 100).toFixed(0)}%</span>
            )}
            {iterationCount !== undefined && iterationCount > 0 && (
              <span>Refined {iterationCount} time{iterationCount === 1 ? '' : 's'} for accuracy</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
