import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  FileText,
  CheckCircle2,
  Cpu,
  Search,
  FileCode,
  Trash2,
} from 'lucide-react';
import { DocumentItem, SearchResultItem } from '../../types';
import { listDocuments, uploadDocument, getDocument, deleteDocument, searchHybrid } from '../../services/api';
import { UploadProgress } from './UploadProgress';
import { ToastStack, ToastMessage } from './Toast';

// How often to re-check a just-uploaded document's status while it moves through
// the pipeline. Short enough to feel live, long enough not to hammer the API for a
// process that realistically takes several seconds to tens of seconds.
const POLL_INTERVAL_MS = 1200;
// Stop polling after this long even if the document never reaches a final state —
// a stuck background task shouldn't poll forever and should surface as a failure
// the user can act on (retry) rather than a progress bar frozen at 70% indefinitely.
const POLL_TIMEOUT_MS = 120_000;

export const DocumentStudio: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [uploadingDoc, setUploadingDoc] = useState<DocumentItem | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const toastIdRef = useRef(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<DocumentItem | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      if (docs.length > 0 && !selectedDoc) {
        setSelectedDoc(docs[0]);
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  };

  const handleDeleteDocument = async (e: React.MouseEvent, doc: DocumentItem) => {
    e.stopPropagation(); // don't trigger the row's onClick (select) as well
    if (!window.confirm(`Delete "${doc.filename}"? This removes it and its embeddings permanently.`)) {
      return;
    }

    setDeletingId(doc.id);
    try {
      await deleteDocument(doc.id);
      setDocuments((prev) => prev.filter((d) => d.id !== doc.id));
      if (selectedDoc?.id === doc.id) {
        setSelectedDoc(null);
      }
    } catch (err: any) {
      console.error('Failed to delete document:', err);
      pushToast('error', `Couldn't delete "${doc.filename}": ${err.message || 'Unknown error'}`);
    } finally {
      setDeletingId(null);
    }
  };

  const pushToast = (kind: ToastMessage['kind'], text: string) => {
    const id = ++toastIdRef.current;
    setToasts((prev) => [...prev, { id, kind, text }]);
  };

  const dismissToast = (id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    // Let the same input accept the same filename again later.
    e.target.value = '';

    let created: DocumentItem;
    try {
      created = await uploadDocument(file);
    } catch (err: any) {
      pushToast('error', `Upload failed: ${err.message || 'Unknown error'}`);
      return;
    }

    setUploadingDoc(created);
    loadDocuments();

    // The upload response only confirms the file reached the server — ingestion
    // (parse -> chunk -> embed) runs afterward in the background, so real progress
    // has to come from polling the document's own status rather than a fixed delay.
    // A fixed delay either finishes the bar before ingestion is actually done (a
    // large PDF can take much longer than any reasonable guess) or leaves the user
    // staring at "done" well after the real work already finished.
    const startedAt = Date.now();
    const poll = async () => {
      let latest: DocumentItem;
      try {
        latest = await getDocument(created.id);
      } catch {
        // A transient poll failure isn't itself the upload failing — keep trying
        // until the timeout, rather than reporting failure on one dropped request.
        if (Date.now() - startedAt < POLL_TIMEOUT_MS) {
          setTimeout(poll, POLL_INTERVAL_MS);
        }
        return;
      }

      setUploadingDoc(latest);

      if (latest.status === 'EMBEDDED') {
        setUploadingDoc(null);
        pushToast('success', `"${latest.filename}" is ready — ${latest.chunk_count ?? 0} sections indexed.`);
        loadDocuments();
        return;
      }
      if (latest.status === 'FAILED') {
        setUploadingDoc(null);
        pushToast('error', `"${latest.filename}" failed to process. Try re-uploading it.`);
        loadDocuments();
        return;
      }
      if (Date.now() - startedAt >= POLL_TIMEOUT_MS) {
        setUploadingDoc(null);
        pushToast('error', `"${latest.filename}" is taking longer than expected. Check back shortly.`);
        return;
      }
      setTimeout(poll, POLL_INTERVAL_MS);
    };

    setTimeout(poll, POLL_INTERVAL_MS);
  };

  const handleHybridSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const res = await searchHybrid(searchQuery, 4);
      setSearchResults(res.results || []);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        overflowY: 'auto',
        padding: '28px 36px',
        display: 'flex',
        flexDirection: 'column',
        gap: '28px',
        background: 'var(--bg-space)',
      }}
    >
      <ToastStack toasts={toasts} onDismiss={dismissToast} />

      {/* Title Header */}
      <div>
        <h1
          style={{
            fontFamily: 'var(--font-heading)',
            fontSize: '24px',
            fontWeight: 700,
            color: '#ffffff',
            marginBottom: '6px',
          }}
        >
          Multimodal Ingestion & Document Studio
        </h1>
        <p style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
          Manage structured document ingestion, Gemini Embedding 2 (256-dim Matryoshka) vector embeddings, and
          test hybrid (Dense + BM25 + Cross-Encoder Rerank) retrieval.
        </p>
      </div>

      {/* Grid: Upload Box + Ingestion Pipeline Status */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
        {/* Drag-and-Drop Upload Area */}
        <div
          className="glass-panel"
          style={{
            padding: '24px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            border: '2px dashed rgba(139, 92, 246, 0.35)',
            background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.04), rgba(6, 182, 212, 0.04))',
            textAlign: 'center',
            position: 'relative',
          }}
        >
          <input
            type="file"
            accept=".pdf,.md,.txt,.png,.jpg,.jpeg"
            onChange={handleFileUpload}
            disabled={!!uploadingDoc}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              opacity: 0,
              cursor: uploadingDoc ? 'default' : 'pointer',
            }}
          />
          {uploadingDoc ? (
            <UploadProgress filename={uploadingDoc.filename} status={uploadingDoc.status} />
          ) : (
            <>
              <div
                style={{
                  width: '52px',
                  height: '52px',
                  borderRadius: '14px',
                  background: 'var(--accent-violet-bg)',
                  color: '#c4b5fd',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '14px',
                }}
              >
                <Upload size={24} />
              </div>
              <div style={{ fontSize: '15px', fontWeight: 600, color: '#ffffff', marginBottom: '6px' }}>
                Drop documents or click to upload
              </div>
              <div style={{ fontSize: '12.5px', color: 'var(--text-muted)' }}>
                Supports PDF (tables + images), Markdown, TXT, and PNG diagrams
              </div>
            </>
          )}
        </div>

        {/* Ingestion Engine Metrics */}
        <div
          className="glass-panel"
          style={{
            padding: '24px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Ingestion Pipeline Config
            </span>
            <span className="badge badge-emerald">ACTIVE</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', margin: '14px 0' }}>
            <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: '8px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Vector Dimension</div>
              <div style={{ fontSize: '16px', fontWeight: 700, color: '#a78bfa', fontFamily: 'var(--font-mono)' }}>
                256-dim MRL
              </div>
            </div>
            <div style={{ background: 'var(--bg-surface)', padding: '10px 12px', borderRadius: '8px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Reranker</div>
              <div style={{ fontSize: '16px', fontWeight: 700, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                Cohere v3.5
              </div>
            </div>
          </div>

          <div style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={14} color="#34d399" />
            <span>Matryoshka representation minimizes storage by 75% with zero quality loss.</span>
          </div>
        </div>
      </div>

      {/* Ingested Documents List */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '16px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FileText size={18} color="#8b5cf6" />
            <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#ffffff' }}>
              Ingested Document Catalog ({documents.length})
            </h3>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {documents.map((doc) => (
            <div
              key={doc.id}
              onClick={() => setSelectedDoc(doc)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 16px',
                background: selectedDoc?.id === doc.id ? 'var(--bg-surface-elevated)' : 'var(--bg-surface)',
                border: selectedDoc?.id === doc.id ? '1px solid var(--accent-violet)' : '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <FileCode size={20} color="#a78bfa" />
                <div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: '#ffffff' }}>
                    {doc.filename}
                  </div>
                  <div style={{ fontSize: '11.5px', color: 'var(--text-muted)' }}>
                    {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : 'Just now'}
                    {' • '}
                    {doc.status === 'FAILED'
                      ? 'Processing failed'
                      : `${doc.chunk_count ?? 0} Chunk${doc.chunk_count === 1 ? '' : 's'}`}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span className={doc.status === 'FAILED' ? 'badge badge-rose' : 'badge badge-emerald'}>
                  <CheckCircle2 size={12} />
                  <span>{doc.status}</span>
                </span>
                <button
                  onClick={(e) => handleDeleteDocument(e, doc)}
                  disabled={deletingId === doc.id}
                  title="Delete document"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '28px',
                    height: '28px',
                    background: 'transparent',
                    border: 'none',
                    borderRadius: 'var(--radius-sm)',
                    cursor: deletingId === doc.id ? 'default' : 'pointer',
                    opacity: deletingId === doc.id ? 0.5 : 1,
                    color: 'var(--text-muted)',
                    transition: 'color 0.15s ease, background 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    if (deletingId !== doc.id) {
                      e.currentTarget.style.color = '#ef4444';
                      e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = 'var(--text-muted)';
                    e.currentTarget.style.background = 'transparent';
                  }}
                >
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive Hybrid Search Test Bench */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
          <Search size={18} color="#06b6d4" />
          <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#ffffff' }}>
            Hybrid Vector + BM25 Retrieval Inspector
          </h3>
        </div>

        <div style={{ display: 'flex', gap: '10px', marginBottom: '16px' }}>
          <input
            type="text"
            className="input-text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleHybridSearch()}
            placeholder="Search across ingested chunks (e.g. 'Matryoshka embeddings', 'RRF fusion score')..."
          />
          <button
            onClick={handleHybridSearch}
            disabled={isSearching || !searchQuery.trim()}
            className="btn btn-primary"
            style={{ padding: '0 20px', flexShrink: 0 }}
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Search Results List */}
        {searchResults.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {searchResults.map((item, idx) => (
              <div
                key={idx}
                style={{
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '14px 16px',
                  borderLeft: '3px solid var(--accent-cyan)',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    marginBottom: '8px',
                    fontSize: '11.5px',
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--text-muted)',
                  }}
                >
                  <span>Chunk: {item.chunk_id}</span>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {item.dense_score && <span style={{ color: '#a78bfa' }}>Dense: {item.dense_score.toFixed(3)}</span>}
                    {item.rerank_score && <span style={{ color: '#34d399' }}>Rerank: {item.rerank_score.toFixed(3)}</span>}
                  </div>
                </div>
                <div style={{ fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                  "{item.text}"
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
