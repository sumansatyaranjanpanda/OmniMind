import React, { useCallback, useEffect, useRef, useState } from 'react';
import { X, Search, Microscope, AlertCircle, Check, ChevronDown } from 'lucide-react';
import { VoiceOrb, OrbState } from './VoiceOrb';
import { VoiceClient, VoiceState } from '../../services/voice';
import { listVoices, VoiceOption } from '../../services/api';
import { CitationItem } from '../../types';

interface VoiceOverlayProps {
  threadId: string | null;
  onClose: () => void;
  /** Called when a spoken turn finishes, so the chat transcript can absorb it. */
  onTurn: (userText: string, agentText: string, citations: CitationItem[]) => void;
}

const STATUS_LABEL: Record<VoiceState, string> = {
  connecting: 'Connecting…',
  listening: 'Listening',
  thinking: 'Looking through your documents…',
  speaking: 'Speaking',
  error: 'Something went wrong',
  closed: 'Call ended',
};

const ORB_STATE: Record<VoiceState, OrbState> = {
  connecting: 'idle',
  listening: 'listening',
  thinking: 'thinking',
  speaking: 'speaking',
  error: 'error',
  closed: 'idle',
};

export const VoiceOverlay: React.FC<VoiceOverlayProps> = ({ threadId, onClose, onTurn }) => {
  const [state, setState] = useState<VoiceState>('connecting');
  const [level, setLevel] = useState(0);
  const [userText, setUserText] = useState('');
  const [agentText, setAgentText] = useState('');
  const [activeTool, setActiveTool] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [voices, setVoices] = useState<VoiceOption[]>([]);
  const [voice, setVoice] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  const clientRef = useRef<VoiceClient | null>(null);
  // The turn callbacks fire from the socket, outside React's render cycle, so
  // the in-flight transcript is mirrored in refs — reading it from state there
  // would capture whatever was current when the client was constructed.
  const userTextRef = useRef('');
  const agentTextRef = useRef('');
  const onTurnRef = useRef(onTurn);
  onTurnRef.current = onTurn;

  // Catalogue first, so the session starts with the user's saved choice rather
  // than connecting on the default and immediately reconnecting.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const catalogue = await listVoices();
      if (cancelled) return;
      setVoices(catalogue.voices);
      setVoice((current) => current ?? localStorage.getItem('omnimind_voice') ?? catalogue.default);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (voice === null) return; // still resolving the catalogue

    const client = new VoiceClient({
      onState: setState,
      onLevel: setLevel,
      onUserTranscript: (text) => {
        userTextRef.current += text;
        setUserText(userTextRef.current);
      },
      onAgentTranscript: (text) => {
        agentTextRef.current += text;
        setAgentText(agentTextRef.current);
      },
      onTurnComplete: (citations) => {
        const user = userTextRef.current.trim();
        const agent = agentTextRef.current.trim();
        if (user && agent) onTurnRef.current(user, agent, citations);
        userTextRef.current = '';
        agentTextRef.current = '';
        setActiveTool(null);
      },
      onToolStart: (tool) => setActiveTool(tool),
      onToolEnd: () => setActiveTool(null),
      onError: setError,
    });

    clientRef.current = client;
    void client.start(threadId, voice);

    return () => {
      client.stop();
      clientRef.current = null;
    };
    // Re-runs when the voice changes, which reconnects with the new voice —
    // the Live session's voice is fixed at connect time, so switching mid-call
    // genuinely requires a new session. threadId is captured deliberately:
    // changing conversations mid-call would silently move audio to another thread.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voice]);

  const handleClose = useCallback(() => {
    clientRef.current?.stop();
    onClose();
  }, [onClose]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Voice conversation"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 200,
        background: 'var(--bg-base)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 'var(--space-6)',
        padding: 'var(--space-8)',
      }}
    >
      {/* Voice picker. Switching reconnects the session, so it's placed away
          from the orb and shows the current choice rather than being a
          hidden setting. */}
      <div style={{ position: 'absolute', top: 'var(--space-5)', left: 'var(--space-5)' }}>
        <button
          onClick={() => setMenuOpen((open) => !open)}
          className="btn"
          aria-haspopup="listbox"
          aria-expanded={menuOpen}
          disabled={voices.length === 0}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 12px',
            borderRadius: 'var(--radius-md)',
            background: 'transparent',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-primary)',
            fontSize: '13px',
          }}
        >
          <span style={{ opacity: 0.55 }}>Voice</span>
          <strong style={{ fontWeight: 550 }}>{voice ?? '…'}</strong>
          <ChevronDown size={14} style={{ opacity: 0.5 }} />
        </button>

        {menuOpen && (
          <ul
            role="listbox"
            style={{
              position: 'absolute',
              top: 'calc(100% + 6px)',
              left: 0,
              minWidth: '220px',
              margin: 0,
              padding: '6px',
              listStyle: 'none',
              background: 'var(--bg-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-lg)',
              boxShadow: 'var(--shadow-lg)',
              maxHeight: '320px',
              overflowY: 'auto',
            }}
          >
            {voices.map((option) => {
              const selected = option.name === voice;
              return (
                <li key={option.name}>
                  <button
                    role="option"
                    aria-selected={selected}
                    onClick={() => {
                      setMenuOpen(false);
                      if (option.name === voice) return;
                      localStorage.setItem('omnimind_voice', option.name);
                      setVoice(option.name);
                    }}
                    style={{
                      width: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '12px',
                      padding: '8px 10px',
                      background: selected ? 'var(--accent-soft)' : 'transparent',
                      border: 'none',
                      borderRadius: 'var(--radius-md)',
                      color: 'var(--text-primary)',
                      fontSize: '13.5px',
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <span>
                      <strong style={{ fontWeight: 550 }}>{option.name}</strong>
                      <span style={{ opacity: 0.5, marginLeft: '8px' }}>{option.character}</span>
                    </span>
                    {selected && <Check size={14} style={{ color: 'var(--accent-text)', flexShrink: 0 }} />}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <button
        onClick={handleClose}
        aria-label="End voice conversation"
        className="btn"
        style={{
          position: 'absolute',
          top: 'var(--space-5)',
          right: 'var(--space-5)',
          padding: '10px',
          borderRadius: 'var(--radius-md)',
          background: 'transparent',
          border: '1px solid var(--border-subtle)',
          color: 'var(--text-primary)',
        }}
      >
        <X size={18} />
      </button>

      <VoiceOrb state={ORB_STATE[state]} level={level} size={280} />

      <div
        aria-live="polite"
        style={{
          fontSize: '13px',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          color: 'var(--text-secondary, var(--text-primary))',
          opacity: 0.7,
          minHeight: '18px',
        }}
      >
        {error ? error : STATUS_LABEL[state]}
      </div>

      {activeTool && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: 'var(--radius-lg)',
            background: 'var(--accent-soft)',
            color: 'var(--accent-text)',
            fontSize: '13px',
          }}
        >
          {activeTool === 'deep_research' ? <Microscope size={14} /> : <Search size={14} />}
          {activeTool === 'deep_research' ? 'Verified research pass' : 'Searching documents'}
        </div>
      )}

      {/* Live transcript. Voice never speaks citation markers aloud, so the
          readable record on screen is what makes a spoken answer auditable. */}
      <div
        style={{
          width: '100%',
          maxWidth: 'var(--measure)',
          minHeight: '120px',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-3)',
        }}
      >
        {userText && (
          <p style={{ margin: 0, textAlign: 'center', fontSize: '15px', opacity: 0.6, color: 'var(--text-primary)' }}>
            "{userText}"
          </p>
        )}
        {agentText && (
          <p style={{ margin: 0, textAlign: 'center', fontSize: '19px', lineHeight: 1.55, color: 'var(--text-primary)' }}>
            {agentText}
          </p>
        )}
      </div>

      {state === 'error' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--danger, #C2413A)', fontSize: '13px' }}>
          <AlertCircle size={14} />
          Close and try again.
        </div>
      )}

      <p style={{ position: 'absolute', bottom: 'var(--space-6)', fontSize: '12px', opacity: 0.45, color: 'var(--text-primary)', margin: 0 }}>
        Just talk — you can interrupt at any time. Press Esc to end.
      </p>
    </div>
  );
};
