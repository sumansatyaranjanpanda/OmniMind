import React, { useEffect, useState } from 'react';
import { FileUp, MessageCircle, Quote, Loader2, FileText } from 'lucide-react';
import { listDocuments } from '../../services/api';
import { DocumentItem } from '../../types';

interface EmptyStateProps {
  onAsk: (query: string) => void;
  onGoToDocuments: () => void;
}

/**
 * First screen of an empty conversation.
 *
 * It branches on whether the account actually has anything to search. A brand-new
 * user pointed at "Summarize my documents" gets an honest but useless "insufficient
 * evidence" answer and no idea why — so with no documents we ask them to add one
 * first, and only offer document questions once there is something to answer from.
 */
export const EmptyState: React.FC<EmptyStateProps> = ({ onAsk, onGoToDocuments }) => {
  const [docs, setDocs] = useState<DocumentItem[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    listDocuments()
      .then((d) => !cancelled && setDocs(d))
      .catch(() => !cancelled && setDocs([]));
    return () => {
      cancelled = true;
    };
  }, []);

  if (docs === null) {
    return (
      <div style={{ margin: 'auto', color: 'var(--text-muted)' }}>
        <Loader2 size={20} className="animate-spin" />
      </div>
    );
  }

  const ready = docs.filter((d) => d.status === 'EMBEDDED');
  const processing = docs.filter((d) => d.status !== 'EMBEDDED' && d.status !== 'FAILED');

  return (
    <div style={{ margin: 'auto', width: '100%', maxWidth: '540px', padding: 'var(--space-8) 0' }}>
      <h2
        style={{
          fontSize: 'var(--text-2xl)',
          fontWeight: 650,
          color: 'var(--text-primary)',
          letterSpacing: '-0.02em',
          margin: '0 0 var(--space-2)',
          textAlign: 'center',
        }}
      >
        {ready.length > 0 ? 'What would you like to know?' : 'Welcome to OmniMind'}
      </h2>
      <p
        style={{
          fontSize: 'var(--text-md)',
          color: 'var(--text-muted)',
          textAlign: 'center',
          margin: '0 0 var(--space-8)',
          lineHeight: 'var(--leading-body)',
        }}
      >
        {ready.length > 0
          ? `Ask anything about your ${ready.length} document${ready.length === 1 ? '' : 's'}. Every answer links back to the exact passage it came from.`
          : 'Add a document, then ask questions about it in plain language. Every answer cites the passage it came from, so you can check it.'}
      </p>

      {ready.length === 0 ? (
        <>
          {/* Three steps, because this genuinely is a sequence — you cannot ask a
              document question before there is a document. */}
          <ol
            style={{
              listStyle: 'none',
              padding: 0,
              margin: '0 0 var(--space-6)',
              display: 'flex',
              flexDirection: 'column',
              gap: 'var(--space-3)',
            }}
          >
            {[
              { icon: FileUp, title: 'Add a document', body: 'PDF, Word, PowerPoint, Excel, CSV or Markdown.' },
              { icon: MessageCircle, title: 'Ask a question', body: 'Plain language — no keywords or syntax needed.' },
              { icon: Quote, title: 'Check the sources', body: 'Every claim carries a citation you can open.' },
            ].map((step, i) => {
              const Icon = step.icon;
              return (
                <li
                  key={i}
                  style={{
                    display: 'flex',
                    gap: 'var(--space-3)',
                    alignItems: 'flex-start',
                    padding: 'var(--space-3)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-subtle)',
                  }}
                >
                  <span
                    style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--accent-soft)',
                      color: 'var(--accent-text)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <Icon size={15} />
                  </span>
                  <span>
                    <span
                      style={{
                        display: 'block',
                        fontSize: 'var(--text-base)',
                        fontWeight: 600,
                        color: 'var(--text-primary)',
                      }}
                    >
                      {step.title}
                    </span>
                    <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-muted)' }}>
                      {step.body}
                    </span>
                  </span>
                </li>
              );
            })}
          </ol>

          <div style={{ display: 'flex', gap: 'var(--space-3)', justifyContent: 'center' }}>
            <button
              onClick={onGoToDocuments}
              className="btn btn-primary"
              style={{ padding: '10px 18px', fontSize: 'var(--text-base)' }}
            >
              <FileUp size={15} />
              <span>Add your first document</span>
            </button>
            <button
              onClick={() => onAsk('What can you help me with?')}
              className="btn btn-secondary"
              style={{ padding: '10px 18px', fontSize: 'var(--text-base)' }}
            >
              Just chat
            </button>
          </div>

          {processing.length > 0 && (
            <p
              style={{
                marginTop: 'var(--space-4)',
                textAlign: 'center',
                fontSize: 'var(--text-sm)',
                color: 'var(--text-muted)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 'var(--space-2)',
              }}
            >
              <Loader2 size={13} className="animate-spin" />
              {processing.length} document{processing.length === 1 ? ' is' : 's are'} still
              processing — you can ask about {processing.length === 1 ? 'it' : 'them'} shortly.
            </p>
          )}
        </>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
          {/* Starters name the user's own files, so the first question is about
              something we can actually answer rather than a generic demo prompt. */}
          {[
            { label: `Summarize ${shortName(ready[0].filename)}`, query: `Summarize the key points of ${ready[0].filename}.` },
            { label: 'Find a specific fact', query: 'What are the most important numbers or dates in my documents?' },
            ...(ready.length > 1
              ? [{ label: 'Compare two documents', query: `Compare ${ready[0].filename} and ${ready[1].filename}. What are the key differences?` }]
              : []),
          ].map((s, i) => (
            <button
              key={i}
              onClick={() => onAsk(s.query)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-3)',
                width: '100%',
                textAlign: 'left',
                padding: 'var(--space-3) var(--space-4)',
                background: 'var(--bg-subtle)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-sans)',
                fontSize: 'var(--text-base)',
                cursor: 'pointer',
                transition: 'background-color var(--dur-1) var(--ease-out), border-color var(--dur-1) var(--ease-out)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'var(--bg-muted)';
                e.currentTarget.style.borderColor = 'var(--border-strong)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'var(--bg-subtle)';
                e.currentTarget.style.borderColor = 'var(--border-subtle)';
              }}
            >
              <FileText size={15} color="var(--text-muted)" />
              <span>{s.label}</span>
            </button>
          ))}

          <button
            onClick={onGoToDocuments}
            className="btn btn-ghost"
            style={{
              alignSelf: 'center',
              marginTop: 'var(--space-3)',
              fontSize: 'var(--text-sm)',
            }}
          >
            Manage documents
          </button>
        </div>
      )}
    </div>
  );
};

/** Trims the extension and truncates, so a starter button doesn't wrap to 3 lines. */
function shortName(filename: string): string {
  const base = filename.replace(/\.[^.]+$/, '');
  return base.length > 34 ? `${base.slice(0, 34)}…` : base;
}
