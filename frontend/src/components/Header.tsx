import React, { useEffect, useRef, useState } from 'react';
import { Sparkles, Sun, Moon, Monitor, LogOut, ChevronDown } from 'lucide-react';
import { getSystemHealth, AuthUser } from '../services/api';
import { SystemHealth } from '../types';

interface HeaderProps {
  user: AuthUser;
  onClearSession: () => void;
  onSignOut: () => void;
}

type ThemeChoice = 'light' | 'dark' | 'system';
const THEME_KEY = 'omnimind_theme';

/** Writes the choice to <html data-theme>. "system" removes the attribute so the
 *  prefers-color-scheme media query decides — the three states the tokens expect. */
function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === 'system') {
    root.removeAttribute('data-theme');
  } else {
    root.setAttribute('data-theme', choice);
  }
}

export const Header: React.FC<HeaderProps> = ({ user, onClearSession, onSignOut }) => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Click-away and Escape both close the menu — a dropdown you can only dismiss
  // by clicking the trigger again feels stuck.
  useEffect(() => {
    if (!menuOpen) return;
    const onPointerDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [menuOpen]);
  const [theme, setTheme] = useState<ThemeChoice>(
    () => (localStorage.getItem(THEME_KEY) as ThemeChoice) || 'system'
  );

  useEffect(() => {
    applyTheme(theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const cycleTheme = () =>
    setTheme((t) => (t === 'system' ? 'light' : t === 'light' ? 'dark' : 'system'));

  const ThemeIcon = theme === 'light' ? Sun : theme === 'dark' ? Moon : Monitor;

  useEffect(() => {
    getSystemHealth().then(setHealth);
    const interval = setInterval(() => {
      getSystemHealth().then(setHealth);
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header
      style={{
        height: 'var(--shell-header)',
        flexShrink: 0,
        borderBottom: '1px solid var(--border-subtle)',
        background: 'var(--bg-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 var(--space-4)',
        zIndex: 20,
      }}
    >
      {/* Brand — flat accent tile, no gradient and no glow */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
        <div
          style={{
            width: '26px',
            height: '26px',
            borderRadius: 'var(--radius-md)',
            background: 'var(--accent)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <Sparkles size={15} color="var(--text-on-accent)" />
        </div>
        <span
          style={{
            fontSize: 'var(--text-lg)',
            fontWeight: 640,
            letterSpacing: '-0.01em',
            color: 'var(--text-primary)',
          }}
        >
          OmniMind
        </span>
      </div>

      {/* Right cluster: status, theme, reset. The three model badges that used to
          live here (gemini-3.7-flash, gemini-embedding-2, Guardrails Shield) named
          internal implementation and meant nothing to someone asking a question. */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
        {/* Status — dot AND label, never colour alone (fails for colour-blind users) */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-2)',
            fontSize: 'var(--text-sm)',
            color: 'var(--text-muted)',
          }}
          title={health?.status === 'HEALTHY' ? 'All services reachable' : 'Checking services'}
        >
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: 'var(--radius-full)',
              backgroundColor:
                health?.status === 'HEALTHY' ? 'var(--success)' : 'var(--warning)',
              flexShrink: 0,
            }}
          />
          <span>{health?.status === 'HEALTHY' ? 'Connected' : 'Connecting'}</span>
        </div>

        <button
          onClick={cycleTheme}
          className="btn btn-ghost"
          style={{ padding: 'var(--space-2)' }}
          title={`Theme: ${theme}. Click to change.`}
          aria-label={`Switch theme, currently ${theme}`}
        >
          <ThemeIcon size={15} />
        </button>

        <button
          onClick={onClearSession}
          className="btn btn-ghost"
          style={{ fontSize: 'var(--text-sm)', padding: 'var(--space-2) var(--space-3)' }}
          title="Start a new conversation"
        >
          New chat
        </button>

        {/* Account menu */}
        <div style={{ position: 'relative' }} ref={menuRef}>
          <button
            onClick={() => setMenuOpen((v) => !v)}
            className="btn btn-ghost"
            style={{
              padding: 'var(--space-1) var(--space-2)',
              gap: 'var(--space-2)',
              fontSize: 'var(--text-sm)',
            }}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
            title={user.email}
          >
            <span
              style={{
                width: '22px',
                height: '22px',
                borderRadius: 'var(--radius-full)',
                background: 'var(--accent-soft)',
                color: 'var(--accent-text)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 'var(--text-xs)',
                fontWeight: 700,
                textTransform: 'uppercase',
                flexShrink: 0,
              }}
            >
              {user.email.charAt(0)}
            </span>
            <ChevronDown size={13} />
          </button>

          {menuOpen && (
            <div
              role="menu"
              className="animate-fade-in-up"
              style={{
                position: 'absolute',
                right: 0,
                top: 'calc(100% + 6px)',
                minWidth: '220px',
                background: 'var(--bg-base)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-lg)',
                boxShadow: 'var(--shadow-lg)',
                padding: 'var(--space-2)',
                zIndex: 'var(--z-drawer)' as React.CSSProperties['zIndex'],
              }}
            >
              <div
                style={{
                  padding: 'var(--space-2) var(--space-3)',
                  borderBottom: '1px solid var(--border-subtle)',
                  marginBottom: 'var(--space-2)',
                }}
              >
                <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                  Signed in as
                </div>
                <div
                  style={{
                    fontSize: 'var(--text-base)',
                    color: 'var(--text-primary)',
                    fontWeight: 500,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {user.email}
                </div>
              </div>
              <button
                role="menuitem"
                onClick={() => {
                  setMenuOpen(false);
                  onSignOut();
                }}
                className="btn btn-ghost"
                style={{
                  width: '100%',
                  justifyContent: 'flex-start',
                  fontSize: 'var(--text-base)',
                  padding: 'var(--space-2) var(--space-3)',
                }}
              >
                <LogOut size={15} />
                <span>Sign out</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
