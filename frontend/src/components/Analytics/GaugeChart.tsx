import React from 'react';

interface GaugeChartProps {
  value: number; // 0 to 1
  label: string;
  target?: number; // e.g. 0.85
  color?: string;
  unit?: string;
}

export const GaugeChart: React.FC<GaugeChartProps> = ({
  value,
  label,
  target = 0.85,
  color = '#8b5cf6',
  unit = '%',
}) => {
  const radius = 58;
  const stroke = 10;
  const normalizedValue = Math.min(Math.max(value, 0), 1);
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - normalizedValue * circumference * 0.75;

  const displayPercent = (normalizedValue * 100).toFixed(0);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        background: 'var(--bg-surface)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-lg)',
        padding: '20px 16px',
        position: 'relative',
      }}
    >
      <div style={{ position: 'relative', width: '130px', height: '110px' }}>
        <svg viewBox="0 0 140 140" style={{ transform: 'rotate(135deg)', width: '100%', height: '100%' }}>
          {/* Background track */}
          <circle
            cx="70"
            cy="70"
            r={radius}
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth={stroke}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * 0.25}
            strokeLinecap="round"
          />

          {/* Filled Arc */}
          <circle
            cx="70"
            cy="70"
            r={radius}
            stroke={color}
            strokeWidth={stroke}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={{
              transition: 'stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
              filter: `drop-shadow(0 0 6px ${color}66)`,
            }}
          />
        </svg>

        {/* Center text value */}
        <div
          style={{
            position: 'absolute',
            top: '40%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center',
          }}
        >
          <div
            style={{
              fontFamily: 'var(--font-heading)',
              fontSize: '24px',
              fontWeight: 800,
              color: '#ffffff',
            }}
          >
            {displayPercent}
            <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>{unit}</span>
          </div>
        </div>
      </div>

      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginTop: '4px' }}>
        {label}
      </div>

      {target && (
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
          Target: &ge; {(target * 100).toFixed(0)}%
        </div>
      )}
    </div>
  );
};
