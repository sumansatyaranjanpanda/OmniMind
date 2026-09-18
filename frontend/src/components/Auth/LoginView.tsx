import React, { useState } from 'react';
import { Sparkles, Loader2, AlertCircle, Eye, EyeOff } from 'lucide-react';
import { login, signup, AuthError } from '../../services/api';

interface LoginViewProps {
  onAuthenticated: () => void;
}

type Mode = 'signin' | 'signup';

// Mirrors api/schemas/auth.py. Checked here only to give immediate feedback —
// the server validates independently and remains the authority.
const MIN_PASSWORD_LENGTH = 8;

function passwordProblem(password: string): string | null {
  if (password.length < MIN_PASSWORD_LENGTH) {
    return `Use at least ${MIN_PASSWORD_LENGTH} characters`;
  }
  if (!/[a-zA-Z]/.test(password)) return 'Include at least one letter';
  if (!/\d/.test(password)) return 'Include at least one number';
  if (new TextEncoder().encode(password).length > 72) {
    return 'That password is too long (max 72 bytes)';
  }
  return null;
}

export const LoginView: React.FC<LoginViewProps> = ({ onAuthenticated }) => {
  const [mode, setMode] = useState<Mode>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // 409 on signup means the address is already registered — the user almost
  // certainly meant to sign in, so offer that instead of leaving them re-reading
  // an error they cannot act on.
  const [emailTaken, setEmailTaken] = useState(false);
  const [busy, setBusy] = useState(false);
  // Only shown once the field has been left, so the rules don't scold mid-typing.
  const [passwordTouched, setPasswordTouched] = useState(false);

  const localPasswordProblem =
    mode === 'signup' && passwordTouched && password ? passwordProblem(password) : null;

  const canSubmit =
    email.trim().length > 3 &&
    email.includes('@') &&
    password.length > 0 &&
    !busy &&
    (mode === 'signin' || passwordProblem(password) === null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;

    setBusy(true);
    setError(null);
    setEmailTaken(false);
    try {
      if (mode === 'signup') {
        await signup(email, password);
      } else {
        await login(email, password);
      }
      onAuthenticated();
    } catch (err) {
      if (err instanceof AuthError) {
        setError(err.message);
        setEmailTaken(mode === 'signup' && err.status === 409);
      } else {
        setError('Something went wrong. Please try again.');
      }
    } finally {
      setBusy(false);
    }
  };

  const switchMode = (next: Mode) => {
    setMode(next);
    setError(null);
    setEmailTaken(false);
    setPasswordTouched(false);
  };

  const inputStyle: React.CSSProperties = {
    width: '100%',
    background: 'var(--bg-base)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-md)',
    padding: '10px 12px',
    color: 'var(--text-primary)',
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--text-base)',
    outline: 'none',
  };

  const labelStyle: React.CSSProperties = {
    display: 'block',
    fontSize: 'var(--text-sm)',
    fontWeight: 600,
    color: 'var(--text-secondary)',
    marginBottom: 'var(--space-2)',
  };

  return (
    <div
      style={{
        minHeight: '100dvh',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-subtle)',
        padding: 'var(--space-6)',
      }}
    >
      <div style={{ width: '100%', maxWidth: '380px' }}>
        {/* Brand */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 'var(--space-3)',
            marginBottom: 'var(--space-6)',
          }}
        >
          <div
            style={{
              width: '34px',
              height: '34px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--accent)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Sparkles size={19} color="var(--text-on-accent)" />
          </div>
          <span
            style={{
              fontSize: 'var(--text-xl)',
              fontWeight: 660,
              letterSpacing: '-0.02em',
              color: 'var(--text-primary)',
            }}
          >
            OmniMind
          </span>
        </div>

        <form
          onSubmit={handleSubmit}
          style={{
            background: 'var(--bg-base)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-lg)',
            padding: 'var(--space-6)',
            display: 'flex',
            flexDirection: 'column',
            gap: 'var(--space-4)',
          }}
        >
          <div>
            <h1
              style={{
                fontSize: 'var(--text-xl)',
                fontWeight: 640,
                color: 'var(--text-primary)',
                margin: 0,
                letterSpacing: '-0.015em',
              }}
            >
              {mode === 'signin' ? 'Sign in' : 'Create your account'}
            </h1>
            <p
              style={{
                fontSize: 'var(--text-base)',
                color: 'var(--text-muted)',
                margin: 'var(--space-1) 0 0',
              }}
            >
              {mode === 'signin'
                ? 'Ask questions about your documents.'
                : 'Upload documents and get answers you can verify.'}
            </p>
          </div>

          {error && (
            <div
              role="alert"
              style={{
                display: 'flex',
                gap: 'var(--space-2)',
                alignItems: 'flex-start',
                background: 'var(--danger-soft)',
                color: 'var(--danger)',
                border: '1px solid transparent',
                borderRadius: 'var(--radius-md)',
                padding: 'var(--space-3)',
                fontSize: 'var(--text-base)',
              }}
            >
              <AlertCircle size={16} style={{ flexShrink: 0, marginTop: '1px' }} />
              <span>
                {error}
                {emailTaken && (
                  <>
                    {' '}
                    <button
                      type="button"
                      onClick={() => switchMode('signin')}
                      style={{
                        background: 'none',
                        border: 'none',
                        padding: 0,
                        color: 'inherit',
                        font: 'inherit',
                        fontWeight: 700,
                        cursor: 'pointer',
                        textDecoration: 'underline',
                      }}
                    >
                      Sign in instead
                    </button>
                  </>
                )}
              </span>
            </div>
          )}

          <div>
            <label htmlFor="email" style={labelStyle}>
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              autoFocus
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
              style={inputStyle}
              onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--border-focus)')}
              onBlur={(e) => (e.currentTarget.style.borderColor = 'var(--border-subtle)')}
            />
          </div>

          <div>
            <label htmlFor="password" style={labelStyle}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <input
                id="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onBlur={(e) => {
                  setPasswordTouched(true);
                  e.currentTarget.style.borderColor = 'var(--border-subtle)';
                }}
                onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--border-focus)')}
                placeholder={mode === 'signup' ? 'At least 8 characters' : 'Your password'}
                style={{ ...inputStyle, paddingRight: '40px' }}
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                style={{
                  position: 'absolute',
                  right: '4px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-muted)',
                  padding: 'var(--space-2)',
                  display: 'flex',
                }}
              >
                {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>

            {mode === 'signup' && (
              <p
                style={{
                  fontSize: 'var(--text-sm)',
                  color: localPasswordProblem ? 'var(--danger)' : 'var(--text-muted)',
                  margin: 'var(--space-2) 0 0',
                }}
              >
                {localPasswordProblem ||
                  'At least 8 characters, including a letter and a number.'}
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={!canSubmit}
            className="btn btn-primary"
            style={{
              width: '100%',
              padding: '10px 16px',
              fontSize: 'var(--text-base)',
              marginTop: 'var(--space-1)',
            }}
          >
            {busy ? (
              <>
                <Loader2 size={15} className="animate-spin" />
                <span>{mode === 'signin' ? 'Signing in…' : 'Creating account…'}</span>
              </>
            ) : (
              <span>{mode === 'signin' ? 'Sign in' : 'Create account'}</span>
            )}
          </button>
        </form>

        <p
          style={{
            textAlign: 'center',
            fontSize: 'var(--text-base)',
            color: 'var(--text-muted)',
            marginTop: 'var(--space-4)',
          }}
        >
          {mode === 'signin' ? "Don't have an account? " : 'Already have an account? '}
          <button
            type="button"
            onClick={() => switchMode(mode === 'signin' ? 'signup' : 'signin')}
            style={{
              background: 'none',
              border: 'none',
              padding: 0,
              color: 'var(--accent-text)',
              fontWeight: 600,
              fontSize: 'var(--text-base)',
              fontFamily: 'var(--font-sans)',
              cursor: 'pointer',
              textDecoration: 'underline',
            }}
          >
            {mode === 'signin' ? 'Create one' : 'Sign in'}
          </button>
        </p>
      </div>
    </div>
  );
};
