import React from 'react';
import { MessageSquare, FileText, Share2, BarChart3 } from 'lucide-react';
import { NavigationTab } from '../types';

interface SidebarProps {
  activeTab: NavigationTab;
  onTabChange: (tab: NavigationTab) => void;
  messageCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onTabChange,
  messageCount,
}) => {
  // Named for what the user does here, not for the machinery behind it. The old
  // labels ("Adaptive Agent / CRAG & SSE Stream", "GraphRAG / Multi-Hop Explorer")
  // described our architecture to someone who only wants to ask a question.
  const navItems = [
    {
      id: 'chat' as NavigationTab,
      label: 'Chat',
      icon: MessageSquare,
      badge: messageCount > 0 ? `${messageCount}` : undefined,
    },
    { id: 'documents' as NavigationTab, label: 'Documents', icon: FileText },
    { id: 'graph' as NavigationTab, label: 'Connections', icon: Share2 },
    { id: 'analytics' as NavigationTab, label: 'Quality', icon: BarChart3 },
  ];

  return (
    <aside
      style={{
        width: 'var(--shell-nav)',
        flexShrink: 0,
        background: 'var(--bg-subtle)',
        borderRight: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        padding: 'var(--space-3) var(--space-2)',
        gap: 'var(--space-1)',
        userSelect: 'none',
        overflowY: 'auto',
      }}
    >
      {navItems.map((item) => {
        const Icon = item.icon;
        const isActive = activeTab === item.id;
        return (
          <button
            key={item.id}
            onClick={() => onTabChange(item.id)}
            aria-current={isActive ? 'page' : undefined}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              width: '100%',
              padding: 'var(--space-2) var(--space-3)',
              borderRadius: 'var(--radius-md)',
              background: isActive ? 'var(--accent-soft)' : 'transparent',
              // A 2px left edge marks the active item — not a gradient, not a glow.
              borderLeft: isActive
                ? '2px solid var(--accent)'
                : '2px solid transparent',
              border: 'none',
              borderLeftWidth: '2px',
              borderLeftStyle: 'solid',
              borderLeftColor: isActive ? 'var(--accent)' : 'transparent',
              color: isActive ? 'var(--accent-text)' : 'var(--text-secondary)',
              cursor: 'pointer',
              textAlign: 'left',
              fontFamily: 'var(--font-sans)',
              fontSize: 'var(--text-base)',
              fontWeight: isActive ? 600 : 500,
              // Named properties only, and no translateY lift — when three
              // components lift on hover, none of them is emphasised.
              transition: 'background-color var(--dur-1) var(--ease-out), color var(--dur-1) var(--ease-out)',
            }}
            onMouseEnter={(e) => {
              if (!isActive) e.currentTarget.style.background = 'var(--bg-muted)';
            }}
            onMouseLeave={(e) => {
              if (!isActive) e.currentTarget.style.background = 'transparent';
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
              <Icon size={16} />
              <span>{item.label}</span>
            </span>

            {item.badge && (
              <span
                style={{
                  fontSize: 'var(--text-xs)',
                  padding: '1px 6px',
                  borderRadius: 'var(--radius-full)',
                  background: isActive ? 'var(--accent)' : 'var(--bg-muted)',
                  color: isActive ? 'var(--text-on-accent)' : 'var(--text-muted)',
                  fontWeight: 600,
                  fontFamily: 'var(--font-mono)',
                  fontVariantNumeric: 'tabular-nums',
                }}
              >
                {item.badge}
              </span>
            )}
          </button>
        );
      })}
    </aside>
  );
};
