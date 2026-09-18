import React from 'react';
import { ShieldCheck, ShieldX, EyeOff } from 'lucide-react';
import { GuardrailStatus } from '../../types';

interface GuardrailBadgeProps {
  status?: GuardrailStatus;
  reason?: string | null;
}

export const GuardrailBadge: React.FC<GuardrailBadgeProps> = ({ status = 'PASSED', reason }) => {
  if (status === 'PASSED') {
    return (
      <div
        className="badge badge-emerald"
        title="Zero prompt injection / jailbreak detected. Input passed enterprise guardrail filter."
      >
        <ShieldCheck size={13} />
        <span>Guardrail: PASSED</span>
      </div>
    );
  }

  if (status === 'PII_MASKED') {
    return (
      <div
        className="badge badge-amber"
        title={reason || 'Sensitive PII detected and securely masked before LLM/search processing.'}
      >
        <EyeOff size={13} />
        <span>Guardrail: PII MASKED</span>
      </div>
    );
  }

  if (status === 'BLOCKED') {
    return (
      <div
        className="badge badge-rose animate-shake"
        title={reason || 'Prompt injection / system override detected and blocked at $0 token cost.'}
      >
        <ShieldX size={13} />
        <span>Guardrail: BLOCKED</span>
      </div>
    );
  }

  return null;
};
