import React, { useCallback, useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { NavigationTab, ChatMessage, SessionStats } from './types';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { LoginView } from './components/Auth/LoginView';
import { ChatView } from './components/Chat/ChatView';
import { DocumentStudio } from './components/Documents/DocumentStudio';
import { GraphExplorer } from './components/Graph/GraphExplorer';
import { EvalDashboard } from './components/Analytics/EvalDashboard';
import {
  AuthUser,
  getCurrentUser,
  logout as apiLogout,
  setSessionEndedHandler,
} from './services/api';

const THREAD_ID_STORAGE_KEY = 'omnimind_thread_id';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavigationTab>('chat');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [threadId, setThreadId] = useState<string | null>(() =>
    localStorage.getItem(THREAD_ID_STORAGE_KEY)
  );
  const [stats, setStats] = useState<SessionStats>({
    totalQueries: 0,
    cacheHits: 0,
    avgFaithfulness: 0.95,
    blockedCount: 0,
    piiMaskedCount: 0,
    intentCounts: {},
    totalTokensEstimated: 0,
  });

  const [user, setUser] = useState<AuthUser | null>(null);
  // Distinguishes "checking for a session" from "no session" — without it the
  // login form flashes on every reload before /auth/me answers.
  const [checkingSession, setCheckingSession] = useState(true);

  const loadUser = useCallback(async () => {
    const current = await getCurrentUser();
    setUser(current);
    setCheckingSession(false);
  }, []);

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  // A 401 can surface from any request, including one fired in the background.
  // Routing them all through here means an expired session lands on the sign-in
  // screen instead of a failed action the user has to interpret.
  useEffect(() => {
    setSessionEndedHandler(() => {
      setUser(null);
      setMessages([]);
      setThreadId(null);
    });
    return () => setSessionEndedHandler(null);
  }, []);

  // The server owns chat history — this only remembers *which* thread to ask for.
  useEffect(() => {
    if (threadId) {
      localStorage.setItem(THREAD_ID_STORAGE_KEY, threadId);
    } else {
      localStorage.removeItem(THREAD_ID_STORAGE_KEY);
    }
  }, [threadId]);

  const handleClearSession = () => {
    setMessages([]);
    setThreadId(null);
  };

  const handleSignOut = () => {
    apiLogout();
    setUser(null);
    setMessages([]);
    setThreadId(null);
    setActiveTab('chat');
  };

  if (checkingSession) {
    return (
      <div
        style={{
          minHeight: '100dvh',
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'var(--bg-subtle)',
          color: 'var(--text-muted)',
        }}
      >
        <Loader2 size={22} className="animate-spin" />
      </div>
    );
  }

  if (!user) {
    return <LoginView onAuthenticated={loadUser} />;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100dvh', overflow: 'hidden' }}>
      <Header
        user={user}
        onClearSession={handleClearSession}
        onSignOut={handleSignOut}
      />

      {/* Main Workspace. `minHeight: 0` is load-bearing: without it a flex child
          refuses to shrink below its content and the inner scroll areas push the
          whole page instead of scrolling inside their own pane. */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden' }}>
        <Sidebar
          activeTab={activeTab}
          onTabChange={setActiveTab}
          messageCount={messages.length}
        />

        <main style={{ flex: 1, display: 'flex', minWidth: 0, overflow: 'hidden' }}>
          {activeTab === 'chat' && (
            <ChatView
              messages={messages}
              setMessages={setMessages}
              threadId={threadId}
              setThreadId={setThreadId}
              stats={stats}
              setStats={setStats}
              onGoToDocuments={() => setActiveTab('documents')}
            />
          )}

          {activeTab === 'documents' && <DocumentStudio />}

          {activeTab === 'graph' && <GraphExplorer />}

          {activeTab === 'analytics' && <EvalDashboard stats={stats} />}
        </main>
      </div>
    </div>
  );
};
