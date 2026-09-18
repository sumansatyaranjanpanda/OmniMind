import React from 'react';
import { X, ExternalLink, FileText, Globe, BookmarkCheck } from 'lucide-react';
import { CitationItem } from '../../types';

interface CitationDrawerProps {
  citation: CitationItem | null;
  onClose: () => void;
}

export const CitationDrawer: React.FC<CitationDrawerProps> = ({ citation, onClose }) => {
  if (!citation) return null;

  const isWeb = citation.source_type === 'web' || !!citation.url;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        right: 0,
        bottom: 0,
        width: '420px',
        maxWidth: '90vw',
        background: 'var(--bg-glass-heavy)',
                borderLeft: '1px solid var(--border-medium)',
        boxShadow: 'var(--shadow-lg)',
        zIndex: 50,
        display: 'flex',
        flexDirection: 'column',
        padding: '24px',
        animation: 'slideInRight 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingBottom: '16px',
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              padding: '6px',
              borderRadius: 'var(--radius-sm)',
              background: isWeb ? 'var(--accent-cyan-bg)' : 'var(--accent-emerald-bg)',
              color: isWeb ? 'var(--info)' : 'var(--success)',
              display: 'flex',
            }}
          >
            {isWeb ? <Globe size={18} /> : <FileText size={18} />}
          </div>
          <div>
            <div style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Source
            </div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Referenced as {citation.marker || '[^1]'} in the answer
            </div>
          </div>
        </div>

        <button
          onClick={onClose}
          className="btn btn-ghost"
          style={{ padding: '6px', borderRadius: '50%' }}
          title="Close Drawer"
        >
          <X size={18} />
        </button>
      </div>

      {/* Metadata Pill List */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '12px',
          padding: '18px 0',
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)', width: '90px' }}>Type:</span>
          <span
            className={`badge ${isWeb ? 'badge-blue' : 'badge-emerald'}`}
            style={{ textTransform: 'uppercase' }}
          >
            {isWeb ? 'Web' : 'Document'}
          </span>
        </div>

        {citation.page !== undefined && citation.page !== null && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', width: '90px' }}>Page Number:</span>
            <span style={{ fontSize: '12.5px', color: 'var(--text-primary)', fontWeight: 500 }}>
              Page {citation.page}
            </span>
          </div>
        )}

        {citation.section && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', width: '90px' }}>Section:</span>
            <span style={{ fontSize: '12.5px', color: 'var(--text-primary)', fontWeight: 500 }}>
              {citation.section}
            </span>
          </div>
        )}

        {citation.url && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', width: '90px' }}>URL:</span>
            <a
              href={citation.url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontSize: '12px',
                color: 'var(--info)',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                textDecoration: 'none',
                wordBreak: 'break-all',
              }}
            >
              <span>{citation.url}</span>
              <ExternalLink size={12} />
            </a>
          </div>
        )}
      </div>

      {/* Text Passage Snippet */}
      <div style={{ padding: '20px 0', flex: 1, overflowY: 'auto' }}>
        <div
          style={{
            fontSize: '11px',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            color: 'var(--text-muted)',
            fontWeight: 700,
            marginBottom: '10px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <BookmarkCheck size={13} color="var(--success)" />
          <span>What it says</span>
        </div>

        <div
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            padding: '16px',
            fontSize: '13.5px',
            lineHeight: 1.65,
            color: 'var(--text-secondary)',
            fontStyle: 'italic',
            borderLeft: '3px solid var(--accent-emerald)',
          }}
        >
          "{citation.text_snippet || 'No excerpt available for this source.'}"
        </div>
      </div>

      {/* Footer Info */}
      <div
        style={{
          paddingTop: '14px',
          borderTop: '1px solid var(--border-subtle)',
          fontSize: '11.5px',
          color: 'var(--success)',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        <BookmarkCheck size={13} />
        <span>Verified as a real source for this answer</span>
      </div>
    </div>
  );
};
