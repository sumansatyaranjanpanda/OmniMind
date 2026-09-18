import React from 'react';
import { DocumentItem } from '../../types';

// Maps the document's real backend status to a percentage and label. Each stage is
// a state ingestion/pipeline.py actually commits to the database as it happens — not
// a guess, so the bar reflects genuine progress rather than a fixed timer.
const STAGE_INFO: Record<DocumentItem['status'], { pct: number; label: string }> = {
  UPLOADED: { pct: 10, label: 'Uploaded — waiting to be read' },
  PARSED: { pct: 40, label: 'Reading document structure' },
  CHUNKED: { pct: 70, label: 'Splitting into searchable sections' },
  EMBEDDED: { pct: 100, label: 'Ready' },
  FAILED: { pct: 100, label: 'Processing failed' },
};

interface UploadProgressProps {
  filename: string;
  status: DocumentItem['status'];
}

export const UploadProgress: React.FC<UploadProgressProps> = ({ filename, status }) => {
  const { pct, label } = STAGE_INFO[status] ?? { pct: 5, label: 'Starting…' };
  const isFailed = status === 'FAILED';

  return (
    <div style={{ width: '100%', maxWidth: '360px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontSize: 'var(--text-sm)',
          marginBottom: 'var(--space-2)',
        }}
      >
        <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{filename}</span>
        <span style={{ color: isFailed ? 'var(--danger)' : 'var(--text-muted)' }}>
          {isFailed ? 'Failed' : `${pct}%`}
        </span>
      </div>
      <div
        style={{
          width: '100%',
          height: '6px',
          borderRadius: 'var(--radius-full)',
          background: 'var(--bg-muted)',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            borderRadius: 'var(--radius-full)',
            background: isFailed ? 'var(--danger)' : 'var(--accent)',
            transition: 'width 0.4s ease',
          }}
        />
      </div>
      <div
        style={{
          fontSize: 'var(--text-xs)',
          color: isFailed ? 'var(--danger)' : 'var(--text-muted)',
          marginTop: 'var(--space-2)',
        }}
      >
        {label}
      </div>
    </div>
  );
};
