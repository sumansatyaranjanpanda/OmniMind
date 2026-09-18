# UI/UX Design Specification — OmniMind Frontend

**Status:** accepted, not yet implemented
**Date:** 2026-09-05
**Applies to:** `frontend/` (React 18 + Vite + TypeScript, per ADR 002)
**Replaces:** the "Cosmic Obsidian" theme currently defined in `frontend/src/index.css`

This document is the single source of truth for how the OmniMind interface looks and
behaves. Implement against it; if you disagree with a decision here, change this file
first and say why, rather than diverging in a component.

---

## Contents

1. [Why this exists](#1-why-this-exists)
2. [Decisions](#2-decisions)
3. [Design tokens](#3-design-tokens)
4. [Token migration map](#4-token-migration-map)
5. [Typography](#5-typography)
6. [Spacing, radius, elevation](#6-spacing-radius-elevation)
7. [Layout](#7-layout)
8. [Component specifications](#8-component-specifications)
9. [Motion](#9-motion)
10. [Turn state machine](#10-turn-state-machine)
11. [Accessibility](#11-accessibility)
12. [Implementation plan](#12-implementation-plan)
13. [Definition of done](#13-definition-of-done)

---

## 1. Why this exists

The current UI has good information architecture and a poor visual system. The
architecture is worth keeping — a reasoning trace, citation chips, a faithfulness score
and a guardrail state are exactly the right things to surface for a citation-first RAG
platform. Four specific things in the skin undermine it:

| Problem | Where | Why it's wrong |
|---|---|---|
| Six full-saturation accent colours | `index.css` — violet, emerald, cyan, amber, rose, blue | When everything is emphasised, nothing is. A product UI needs one accent plus semantic state colours. |
| Glow shadows | `--shadow-glow-violet: 0 0 20px …` | Outer glow is a games/crypto-landing device. It reads as "demo" instantly. |
| Glassmorphism | `backdrop-filter: blur(16px)` in `.glass-panel`, `Header.tsx` | A 2021 trend that costs GPU on every scroll and reduces text contrast. |
| Permanent animation | `pulseGlow 2s infinite`, `translateY(-1px)` on three components | Motion with no state change is decoration the eye keeps fighting. |

And one measurable defect, which is not a matter of taste:

> **`ChatView.tsx` has no max-width on the answer column.** The message list is
> `padding: '24px 32px 140px 32px'` inside an uncapped flex column. On a 1920px display
> answer text runs ≈230 characters per line. Comfortable reading is 60–75. Past ~90 the
> eye loses its place on the line return, and long RAG answers are where that hurts most.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Light theme: cobalt on white.** Accent `#2C5FF6`, ground `#FFFFFF`. | Enterprise-tool authority. This is the register a platform that searches company documents needs to speak in. Reference: Glean. |
| D2 | **Dark theme: clay on warm charcoal.** Accent `#E08A5F`, ground `#262624`. | Not pure black. Warm charcoal is materially easier to sit in for a long session than `#06080f`, and it differentiates from every cold blue-black AI tool. Reference: Claude. |
| D3 | **The accent changes hue between themes.** | Cobalt holds authority on white and goes muddy on warm charcoal; clay carries the dark theme instead. The accent is a *role*, not a fixed hex. |
| D4 | **One accent + four semantic colours.** Success, warning, danger, info — used only for state, never decoration. | Restores hierarchy. |
| D5 | **Layout 3 — evidence workspace**, collapsing to Layout 1 below 1100px. | Sources stay permanently on screen. The product's claim is verifiable evidence; the layout should argue it. |
| D6 | **Answer column capped at 68ch.** | Reading measure. Highest-impact single fix in the codebase. |
| D7 | **Four-step motion scale.** 120 / 180 / 240 / 280ms. | Replaces six ad-hoc durations and four ad-hoc easings. |
| D8 | **One type family.** Drop Plus Jakarta Sans; keep one sans + one mono. | Two near-identical sans faces is a tell, not a system. |
| D9 | **Theme is user-switchable**, defaulting to system preference. | Currently there is only a dark theme and no toggle. |

**Fallback if D5 is too much work for the first pass:** ship Layout 1 (sources inline
under the answer). It uses the same tokens and the same components; only the shell grid
differs. Do not ship Layout 2 — it optimises for a single-screen product, and OmniMind has
four views, two of which are dense data.

---

## 3. Design tokens

Replace the entire `:root` block at the top of `frontend/src/index.css` with this.
Every colour is defined at the `:root` level for light; dark redefines **only** the token
values, never component rules.

```css
/* ==========================================================================
   OmniMind Design Tokens
   Light: cobalt on white.  Dark: clay on warm charcoal.
   Component rules must never reference a raw hex — only these tokens.
   ========================================================================== */

:root {
  /* ── Surfaces ───────────────────────────────────────────── */
  --bg-base:        #FFFFFF;   /* app ground, content areas */
  --bg-subtle:      #F7F8FA;   /* panels, sidebar, cards */
  --bg-muted:       #EEF1F5;   /* input fields, hover fills, skeletons */
  --bg-inset:       #E7EBF1;   /* pressed states, code blocks */

  /* ── Borders ────────────────────────────────────────────── */
  --border-subtle:  #E3E7EC;
  --border-strong:  #CBD3DD;
  --border-focus:   #2C5FF6;

  /* ── Text ───────────────────────────────────────────────── */
  --text-primary:   #0E1116;
  --text-secondary: #414A57;
  --text-muted:     #5B6472;
  --text-disabled:  #98A1AD;
  --text-on-accent: #FFFFFF;

  /* ── Accent (single) ────────────────────────────────────── */
  --accent:         #2C5FF6;
  --accent-hover:   #1B47C9;
  --accent-soft:    #EAF0FF;   /* tinted background for chips, active nav */
  --accent-text:    #1B47C9;   /* accent used as text on a light ground */

  /* ── Semantic state ─────────────────────────────────────── */
  --success:        #137A4B;
  --success-soft:   #E4F4EC;
  --warning:        #8A5D00;
  --warning-soft:   #FCF1DC;
  --danger:         #B3341F;
  --danger-soft:    #FCEAE6;
  --info:           #1B5E8A;
  --info-soft:      #E5F1F8;

  /* ── Elevation (no glow, ever) ──────────────────────────── */
  --shadow-sm:      0 1px 2px rgba(14, 17, 22, 0.06);
  --shadow-md:      0 1px 2px rgba(14, 17, 22, 0.06), 0 4px 12px rgba(14, 17, 22, 0.05);
  --shadow-lg:      0 2px 4px rgba(14, 17, 22, 0.06), 0 12px 32px rgba(14, 17, 22, 0.08);

  /* ── Typography ─────────────────────────────────────────── */
  --font-sans: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  --font-mono: "JetBrains Mono", "Cascadia Mono", Consolas, ui-monospace, monospace;

  --text-xs:   11px;
  --text-sm:   12.5px;
  --text-base: 14px;
  --text-md:   15px;
  --text-lg:   17px;
  --text-xl:   20px;
  --text-2xl:  26px;

  --leading-tight: 1.25;
  --leading-body:  1.6;
  --leading-loose: 1.75;

  /* ── Spacing (4px base) ─────────────────────────────────── */
  --space-1: 4px;   --space-2: 8px;   --space-3: 12px;  --space-4: 16px;
  --space-5: 20px;  --space-6: 24px;  --space-8: 32px;  --space-10: 40px;
  --space-12: 48px; --space-16: 64px;

  /* ── Radius ─────────────────────────────────────────────── */
  --radius-sm:   4px;
  --radius-md:   6px;
  --radius-lg:   10px;
  --radius-full: 9999px;

  /* ── Layout constants ───────────────────────────────────── */
  --shell-header:   48px;
  --shell-nav:      220px;
  --shell-nav-rail: 56px;
  --shell-evidence: 320px;
  --measure:        68ch;    /* answer column cap — see D6 */

  /* ── Motion ─────────────────────────────────────────────── */
  --dur-1: 120ms;   /* hover, press, focus, chip select */
  --dur-2: 180ms;   /* element enters or leaves */
  --dur-3: 240ms;   /* expand / collapse */
  --dur-4: 280ms;   /* panel, modal, drawer */

  --ease-out:    cubic-bezier(0, 0, 0.2, 1);      /* entering */
  --ease-in:     cubic-bezier(0.4, 0, 1, 1);      /* leaving */
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);    /* moving on screen */

  --z-base: 0; --z-panel: 10; --z-header: 20; --z-drawer: 30; --z-modal: 40; --z-toast: 50;
}

/* Dark: system preference, unless the user explicitly chose light */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg-base:        #262624;
    --bg-subtle:      #2E2E2B;
    --bg-muted:       #37372F;
    --bg-inset:       #1F1F1D;

    --border-subtle:  #413F3A;
    --border-strong:  #55524A;
    --border-focus:   #E08A5F;

    --text-primary:   #F0EEE6;
    --text-secondary: #C9C4B8;
    --text-muted:     #A9A49A;
    --text-disabled:  #75716A;
    --text-on-accent: #241C17;

    --accent:         #E08A5F;
    --accent-hover:   #EC9E77;
    --accent-soft:    #3A2E27;
    --accent-text:    #F0A87E;

    --success:        #7CC49B;
    --success-soft:   #2A362E;
    --warning:        #D9B166;
    --warning-soft:   #3A3327;
    --danger:         #E28A7A;
    --danger-soft:    #3B2A26;
    --info:           #86B9D6;
    --info-soft:      #26333A;

    --shadow-sm:      0 1px 2px rgba(0, 0, 0, 0.35);
    --shadow-md:      0 1px 2px rgba(0, 0, 0, 0.35), 0 4px 12px rgba(0, 0, 0, 0.25);
    --shadow-lg:      0 2px 4px rgba(0, 0, 0, 0.4),  0 12px 32px rgba(0, 0, 0, 0.35);
  }
}

/* Dark: explicit user choice — must win over a light OS setting */
:root[data-theme="dark"] {
  /* identical body to the media block above; keep the two in sync */
}
```

**Three rules that keep this working:**

1. A colour must never be defined only inside the media query or the `[data-theme]`
   block. The bare `:root` carries the complete light palette; the others redefine
   token *values* only.
2. No component rule may contain a raw hex. If you need a colour that isn't a token,
   add the token here first.
3. `body` must set `background: var(--bg-base)` explicitly. A transparent body borrows
   the host's ground and produces one theme's text on the other theme's background.

### Contrast (measured, WCAG 2.1)

| Pair | Ratio | Requirement | Result |
|---|---|---|---|
| `--text-primary` on `--bg-base` (light) | 18.1:1 | 4.5:1 | pass |
| `--text-muted` on `--bg-base` (light) | 6.0:1 | 4.5:1 | pass |
| `--accent` on `--bg-base` (light) | 5.2:1 | 4.5:1 | pass |
| `--text-on-accent` on `--accent` (light) | 5.2:1 | 4.5:1 | pass |
| `--text-primary` on `--bg-base` (dark) | 13.1:1 | 4.5:1 | pass |
| `--accent` on `--bg-base` (dark) | 5.8:1 | 4.5:1 | pass |
| `--text-on-accent` on `--accent` (dark) | 6.4:1 | 4.5:1 | pass |

Re-measure any token you change. `--text-disabled` is intentionally below 4.5:1 — it must
only ever be used on genuinely disabled controls, which are exempt.

---

## 4. Token migration map

There are **218 inline `style={{}}` blocks** across `frontend/src/components/`, and many
already reference token names by string (`'var(--border-subtle)'`). A rename therefore
touches components whether you want it to or not.

**Strategy: alias, migrate, delete.** Append this compatibility block immediately after
the new tokens so the app never breaks mid-migration, then remove it in Phase 5.

```css
/* TEMPORARY — remove once no component references these. See §12 Phase 5. */
:root {
  --bg-space:            var(--bg-base);
  --bg-deep:             var(--bg-base);
  --bg-surface:          var(--bg-subtle);
  --bg-surface-elevated: var(--bg-muted);
  --bg-surface-active:   var(--bg-inset);
  --bg-glass:            var(--bg-subtle);
  --bg-glass-heavy:      var(--bg-subtle);
  --border-medium:       var(--border-strong);
  --border-bright:       var(--border-strong);
  --accent-violet:       var(--accent);
  --accent-violet-bg:    var(--accent-soft);
  --accent-emerald:      var(--success);
  --accent-emerald-bg:   var(--success-soft);
  --accent-amber:        var(--warning);
  --accent-amber-bg:     var(--warning-soft);
  --accent-rose:         var(--danger);
  --accent-rose-bg:      var(--danger-soft);
  --accent-blue:         var(--info);
  --accent-blue-bg:      var(--info-soft);
  --accent-cyan:         var(--info);
  --text-dim:            var(--text-disabled);
  --font-heading:        var(--font-sans);
  --radius-xl:           var(--radius-lg);
  --shadow-glow-violet:  none;
  --shadow-glow-emerald: none;
}
```

Note what the aliases do: the six neon accents collapse onto one accent plus four
semantic colours, and the glow shadows become `none`. The visual cleanup happens the
moment the tokens land, before a single component is edited.

**Hard-coded values the aliases can't catch** — these need manual edits:

| File | Value | Replace with |
|---|---|---|
| `Header.tsx` | `linear-gradient(135deg, #8B5CF6, #06B6D4)` on the logo tile | `var(--accent)` flat fill |
| `Header.tsx` | `linear-gradient(to right, #ffffff, #cbd5e1)` + `WebkitBackgroundClip: 'text'` | `color: var(--text-primary)` |
| `Header.tsx` | `<Sparkles color="#ffffff" />` | `color="var(--text-on-accent)"` |
| `index.css` | `.btn-primary` gradient fill | `background: var(--accent)` |
| `index.css` | `.badge-*` literal colours (`#34d399`, `#fbbf24`, `#f87171`, `#60a5fa`, `#a78bfa`) | matching semantic token |
| `index.css` | `.citation-ref` emerald colours | `--accent-soft` / `--accent-text` |

Find every remaining literal with:

```bash
grep -rnE "#[0-9a-fA-F]{3,8}\b" frontend/src --include=*.tsx --include=*.css
```

---

## 5. Typography

One sans, one mono. `Plus Jakarta Sans` is removed (D8).

| Role | Token | Size | Weight | Line height | Used for |
|---|---|---|---|---|---|
| Display | `--text-2xl` | 26px | 650 | 1.25 | empty-state headline only |
| Title | `--text-xl` | 20px | 640 | 1.3 | view titles |
| Section | `--text-lg` | 17px | 620 | 1.35 | card headers, document names |
| Body | `--text-md` | 15px | 400 | 1.6 | answer text, prose |
| UI | `--text-base` | 14px | 500 | 1.5 | buttons, nav, labels |
| Small | `--text-sm` | 12.5px | 400 | 1.5 | metadata, captions |
| Micro | `--text-xs` | 11px | 500 | 1.4 | badges, citation markers |

**Mono is for machine output only** — route history, chunk IDs, scores, model names,
file paths, code. Never for prose, and never for a label just because it looks technical.

**Numeric columns** (eval dashboard, chunk counts, scores) get
`font-variant-numeric: tabular-nums` so digits align.

Fonts are self-hosted or system. Do not add a font CDN link — it fails silently behind a
strict CSP and you get an unnoticed fallback.

---

## 6. Spacing, radius, elevation

**Spacing** is the 4px scale in §3. Lay out sibling groups with flex/grid + `gap`, never
per-element margins — collapsed and doubled margins are the most common layout bug in
this codebase's style.

**Radius:**

| Token | Value | Applies to |
|---|---|---|
| `--radius-sm` | 4px | badges, citation markers, inline chips |
| `--radius-md` | 6px | buttons, inputs, cards, source rows |
| `--radius-lg` | 10px | panels, modals, the composer |
| `--radius-full` | — | avatars, status dots, pills |

`--radius-xl: 20px` is removed. At that size a card starts reading as a mobile widget.

**Elevation:** three shadows, no glow. Prefer a `1px` border over a shadow for anything
that isn't genuinely floating. Panels and cards that sit *in* the layout get a border;
only drawers, modals and popovers get `--shadow-lg`.

---

## 7. Layout

### 7.1 Shell (Layout 3 — evidence workspace)

```
┌──────────────────────────────────────────────────────────────┐
│ header  48px — brand · health · theme toggle · clear session │
├────────────┬────────────────────────────────┬────────────────┤
│ nav        │ workspace                      │ evidence       │
│ 220px      │ 1fr, content capped at 68ch    │ 320px          │
│            │                                │                │
│ · Chat     │   answer + composer            │  sources       │
│ · Documents│                                │  route trace   │
│ · Graph    │                                │  faithfulness  │
│ · Evaluation                                │  guardrail     │
└────────────┴────────────────────────────────┴────────────────┘
```

Implement as CSS grid on the shell, not nested flex:

```css
.shell {
  display: grid;
  grid-template-columns: var(--shell-nav) minmax(0, 1fr) var(--shell-evidence);
  grid-template-rows: var(--shell-header) minmax(0, 1fr);
  height: 100dvh;
}
```

`minmax(0, 1fr)` matters — without the `0` minimum, a wide child (the graph canvas, a code
block) forces the whole grid to overflow instead of scrolling inside its own pane.

### 7.2 Dimensions

| Element | Value | Change from current |
|---|---|---|
| Header | 48px | was 64px |
| Nav | 220px | was 260px |
| Nav (rail mode) | 56px, icon-only + tooltip | new |
| Evidence panel | 320px | new (was an overlay drawer) |
| Answer column | `max-width: var(--measure)` (68ch), centred | **was uncapped — the bug in §1** |
| Composer | sticky bottom, `padding-block: var(--space-4)` | was a hard-coded 140px reserve |

### 7.3 Breakpoints

| Width | Behaviour |
|---|---|
| ≥ 1400px | Full Layout 3. Evidence panel 360px. |
| 1100–1399px | Full Layout 3. Evidence panel 320px. |
| 768–1099px | Evidence panel collapses to an overlay drawer (Layout 1 behaviour); nav becomes a 56px rail. |
| < 768px | Single column. Nav becomes a bottom bar. Evidence opens full-screen. |

The 100dvh unit (not `100vh`) avoids the mobile browser chrome bug where the composer
sits under the URL bar.

### 7.4 Per-view

| View | Layout |
|---|---|
| **Chat** | Answer column capped at 68ch. Evidence panel live. Composer sticky. |
| **Documents** | Full workspace width — a table, not a card grid. Columns: name, type, status, chunks, uploaded, actions. Evidence panel shows the selected document's chunk preview. |
| **Graph** | Full-bleed canvas; the 68ch cap does not apply. Evidence panel shows the selected node's entity detail and source chunks. |
| **Evaluation** | Two-column metric grid above a run table. Evidence panel hidden — nothing to cite. |

When a view hides the evidence panel, the grid drops to two columns rather than leaving
an empty pane.

---

## 8. Component specifications

### 8.1 `Header.tsx` — 48px

Brand mark (flat `--accent` tile, no gradient, no glow), product name in
`--text-primary`, then right-aligned: health indicator, theme toggle, clear-session.
Remove `backdropFilter: blur(20px)` — a solid `--bg-subtle` with a bottom border.

Health indicator: an 8px dot in `--success` / `--warning` / `--danger` plus a text label.
Colour alone is not a status — it fails for colour-blind users.

### 8.2 `Sidebar.tsx` — 220px

Four items, icon + label. Active item: `--accent-soft` background, `--accent-text` text,
and a 2px `--accent` left edge. Not a gradient, not a glow.

Remove the hover `translateY`. Hover is `--bg-muted`, 120ms.

### 8.3 `MessageBubble.tsx`

- **User turn:** right-aligned, `--bg-muted` fill, `--radius-lg`, max 80% of the column.
- **Assistant turn:** no bubble. Full column width, `--bg-base`, body type. The answer is
  the page, not a chat balloon — it can be several hundred words and a bubble makes long
  prose harder to read.
- Enters with `fadeInUp`, retimed to `--dur-2` and 4px of travel (was 250ms / 10px).

### 8.4 `citation-ref` (inline marker)

11px mono, `--radius-sm`, `--accent-soft` background, `--accent-text` text, superscript
alignment. Hover: `--accent` background, `--text-on-accent` text, 120ms, **no lift, no
glow**. Click scrolls the evidence panel to that source and applies a 1.2s
`--accent-soft` highlight to the row.

### 8.5 Evidence panel (replaces `CitationDrawer.tsx`)

Persistent right pane in Layout 3; an overlay drawer below 1100px. Sections, in order:

1. **Sources** — numbered rows: `[n]`, filename, page/section, relevance score. Clicking
   opens the original.
2. **Route trace** — collapsed summary line (`4 steps · 1.8s · VERIFIED`), expandable to
   the node list. Collapsed by default.
3. **Faithfulness** — the score as a value plus a bar, with the 0.85 threshold marked.
4. **Guardrail** — only rendered when the status is not `PASSED`.

Sources populate on the `citations` SSE event, which — measured against the live
server — arrives *after* the answer text, not before it:

```
0.65s  route  STARTING / GUARDRAIL_EVALUATED
1.71s  route  CACHE_MISS
3.85s  route  CONTEXT_REWRITTEN
5.09s  route  ANALYZED
9.75s  route  RETRIEVED / FUSED
39.80s        first token
39.81s        citations  (n=1)
40.88s        complete
```

The backend emits `citations` from the `astream` update for the synthesizer node,
which only fires once that node has fully returned — so every token is already on the
wire by then. Design the panel to show a **sources skeleton** during the wait and fill
it at the end; do not build an interaction that assumes sources are available while
text is still arriving. If sources-before-text is wanted, that is a backend change
(emit citations from inside the synthesizer once evidence is fused), not a frontend one.

### 8.6 `ReasoningTrace.tsx`

Collapsed by default. Summary line shows step count, elapsed time, and verification
status. While streaming, the summary updates live with the current node. Expand/collapse
uses `--dur-3` with `--ease-in-out`.

This shows *status*, never model reasoning — the route list is a sequence of node names,
which keeps it on the right side of the "never expose raw chain-of-thought" rule in
`AGENTS.md`.

### 8.7 `GuardrailBadge.tsx`

Three states, using semantic tokens: `PASSED` → `--success`, `PII_MASKED` → `--warning`,
`BLOCKED` → `--danger`. Icon + label; never colour alone.

**Remove `animate-shake` from the `BLOCKED` state.** A shake is a physical "no" gesture,
right for a rejected password and wrong for a policy explanation. Replace with a
180ms fade-in of a message panel with a `--danger` left border that states what was
blocked and what the user can do instead.

### 8.8 `ThinkingIndicator.tsx`

Two modes:

- **Pre-token:** four skeleton lines at 92% / 100% / 74% / 45% width with a 1.4s shimmer.
  Shows the shape of the answer so the layout doesn't jump when text lands.
- **Streaming:** three staggered dots (`--dur` 1.2s) beside the current node label, plus a
  2px caret at the end of the streamed text.

Dots are `--text-muted`, not the accent. A loading state is not an emphasis.

### 8.9 Composer

`--radius-lg`, `--bg-subtle`, 1px border. Focus: `--border-focus` plus a 3px
`--accent-soft` ring, 120ms. Auto-grows to a 6-line maximum then scrolls internally.
During generation the send button becomes **Stop**, which must actually abort the SSE
reader — a stop button that only hides the spinner is a lie.

### 8.10 Buttons

| Variant | Rest | Hover | Active |
|---|---|---|---|
| Primary | `--accent` / `--text-on-accent` | `--accent-hover` | `inset 0 1px 3px rgba(0,0,0,.25)` |
| Secondary | `--bg-subtle`, 1px `--border-subtle` | `--bg-muted` | inset shadow |
| Ghost | transparent, `--text-secondary` | `--bg-muted` | inset shadow |
| Danger | `--danger-soft` / `--danger` | `--danger` / `--text-on-accent` | inset shadow |

All transitions: `background-color var(--dur-1) var(--ease-out)`. No gradient fills, no
`translateY`, no `transition: all`.

---

## 9. Motion

### 9.1 The rule

> **Animation may do exactly two things: show that something changed state, or show
> where something came from.** Anything else is decoration and gets cut.

Apply this test to every animation before adding it. `pulseGlow` fails it — nothing
changed and nothing moved.

### 9.2 Scale

| Token | Duration | Easing | Applies to |
|---|---|---|---|
| `--dur-1` | 120ms | `--ease-out` | hover, press, focus ring, chip select, citation hover |
| `--dur-2` | 180ms | `--ease-out` | message enters, badge appears, tooltip, error panel |
| `--dur-3` | 240ms | `--ease-in-out` | reasoning trace expand/collapse, accordion |
| `--dur-4` | 280ms | `--ease-out` | evidence panel, modal, document drawer |

Easing by direction: ease-out for entering, ease-in for leaving, ease-in-out for something
moving while staying on screen, linear only for spinners and progress bars.

Nothing below 80ms — the change isn't perceived as motion, it reads as a glitch. Anything
the user triggers dozens of times a session stays at `--dur-1`.

### 9.3 Delete

| Animation | Why |
|---|---|
| `pulseGlow 2s infinite` | Permanent motion, no state change. |
| `translateY(-1px)` on `.btn`, `.preset-chip`, `.citation-ref` | Three components lifting means none is emphasised. |
| `transition: all` (everywhere) | Name the properties; `all` animates layout properties you didn't intend and costs frames. |
| `redShake` on guardrail block | See §8.7. |
| Glow `box-shadow` on citation hover | A background change already says "clickable". |
| `.glass-panel` / `.glass-card` `backdrop-filter` | GPU cost per scroll, lower contrast, dated. |

### 9.4 Keep, retimed

| Animation | Change |
|---|---|
| `cursorBlink` | Keep as is. Encodes live generation; stops when generation stops. |
| `pulseDot` | Keep the 1.2s stagger. Recolour to `--text-muted`. |
| `fadeInUp` | 250ms → `--dur-2`; travel 10px → 4px. |
| `slideInRight` | 300ms → `--dur-4`; easing → `--ease-out`. |

### 9.5 Reduced motion

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

The streaming caret and skeleton shimmer must also be suppressed in JS where they're
driven by timers, not only CSS — check
`window.matchMedia('(prefers-reduced-motion: reduce)')` before starting an interval.

---

## 10. Turn state machine

Every chat turn passes through these. Each needs a distinct visual state; today several
of them look identical.

| State | Trigger | UI |
|---|---|---|
| `idle` | — | Composer enabled, empty-state prompt suggestions if no messages. |
| `submitted` | user sends | Input frozen, user message enters (`--dur-2`), skeleton appears in the answer slot. |
| `routing` | `route_update` SSE | **The dominant visual state — see the note below.** Reasoning trace summary updates live with the current node. Skeleton still showing. |
| `streaming` | first `token` SSE | Skeleton replaced by text + caret. Send button becomes Stop. |
| `evidence` | `citations` SSE | Evidence panel fills. Arrives *after* the text, not before — see §8.5. Sources fade in staggered 40ms apart. |
| `verified` | `complete`, status `VERIFIED` | Caret removed, faithfulness score animates to value over `--dur-3`, trace collapses to summary. |
| `unverified` | `complete`, status `PARTIALLY_VERIFIED` / `INSUFFICIENT_EVIDENCE` | Same, plus a `--warning` banner above the answer naming the score and that retries were exhausted. **This state must be visible** — the critic gate is decorative if the UI hides a failed verification. |
| `blocked` | `guardrail_status: BLOCKED` | No answer body. A `--danger`-bordered panel fades in (`--dur-2`) stating what was blocked. No shake. |
| `cached` | `cache_hit: true` | Answer renders immediately (~1.3s) with a `--info` "from cache" chip. Do not fake a stream for a cached answer. **Cached answers arrive with `citations: []` while the text still contains `[^n]` markers** — render those markers as plain text, not clickable chips, or they are dead controls. See the note below. |
| `error` | network / SSE failure | Inline panel with the failure and a Retry button. Never a bare spinner that stops. |

### 10.1 Two measured facts that change the design

**The skeleton is the experience, not a flicker.** Time-to-first-token is essentially
identical to total time — 21.3s vs 22.7s on one measured query, 39.8s vs 40.9s on
another. Roughly 95% of the wait happens *before* any text exists, then the answer lands
almost at once. Streaming is real but delivers close to zero perceived benefit today.

The consequence: `routing` is the state the user actually spends their time in, so the
live route trace is not a nice-to-have detail — it is the loading experience, and the
only thing standing between the user and a 20-second blank screen. Give it real design
attention: name each stage in plain language ("Searching your documents", "Reading 6
passages", "Writing the answer"), not node identifiers.

**Cached answers have markers but no citation data.** Verified against the live server:
`cache_hit=true`, `citations: []`, and the answer body still containing `[^1]`. The
semantic cache stores the answer text but `finalize_cached` never restores the citation
list, so every marker in a cached answer points at nothing. Until that is fixed
server-side, the UI must not render markers as interactive when `citations` is empty —
a chip that does nothing when clicked is worse than plain text.

---

## 11. Accessibility

- **Contrast:** every text/background pair meets 4.5:1, UI components 3:1. Table in §3.
- **Focus:** `:focus-visible` gets a 2px `--border-focus` outline at 2px offset, on every
  interactive element. Never `outline: none` without a replacement.
- **Colour is never the only signal.** Guardrail states, health, and verification status
  all carry an icon or text label alongside the colour.
- **Keyboard:** the composer is reachable with `/`, `Esc` closes the evidence drawer,
  `Tab` order follows visual order. The nav rail's icon-only mode needs `aria-label` on
  every item.
- **Live regions:** the streaming answer container is `aria-live="polite"` so a screen
  reader announces the completed answer — not every token.
- **Motion:** honoured per §9.5.
- **Targets:** interactive elements are at least 32×32px; 44×44px below 768px.

---

## 12. Implementation plan

Ordered so the app is shippable after every phase. Verify before moving on.

### Phase 1 — Tokens (largest visual win, smallest diff)

1. Replace the `:root` block in `frontend/src/index.css` with §3.
2. Append the compatibility aliases from §4.
3. Add `body { background: var(--bg-base); color: var(--text-primary); }`.

**Verify:** `npm run dev`. The app is now light-themed, single-accent, glow-free, without
a single component edit. Nothing should be unreadable; anything that is means a component
holds a hard-coded colour — fix it there.

### Phase 2 — The layout bug

4. Cap the answer column: wrap the message list in a `max-width: var(--measure); margin-inline: auto`.
5. Replace the 140px composer reserve with a sticky composer.

**Verify:** open a long answer at 1920px. Lines should be ~68 characters, not ~230.

### Phase 3 — Motion cleanup

6. Delete everything in §9.3 from `index.css`.
7. Retime the four animations in §9.4.
8. Replace every `transition: all` with named properties.

**Verify:** `grep -n "transition: all\|glow\|backdrop-filter\|translateY(-1px)" frontend/src/index.css` returns nothing.

### Phase 4 — Shell and evidence panel

9. Convert `App.tsx` from nested flex to the CSS grid shell in §7.1.
10. Header 64 → 48px, sidebar 260 → 220px.
11. Rewrite `CitationDrawer.tsx` as the persistent evidence panel (§8.5), with the
    overlay fallback below 1100px.
12. Add the breakpoint rules from §7.3.

**Verify:** resize from 1920 to 375px. No horizontal body scroll at any width; the panel
collapses at 1100px; the nav becomes a bottom bar under 768px.

### Phase 5 — Component pass and alias removal

13. Work through §8 component by component, replacing aliased tokens with real ones.
14. Fix the hard-coded values listed in §4.
15. Delete the compatibility alias block.
16. Add the theme toggle (D9) writing `data-theme` to `documentElement` and `localStorage`.

**Verify:** `grep -rnE "#[0-9a-fA-F]{3,8}\b" frontend/src --include=*.tsx --include=*.css`
returns only the token definitions in `index.css`. Toggle between themes on every view.

### Phase 6 — States

17. Implement the full state machine in §10, particularly `unverified` and `cached`.
18. Implement the skeleton and the working Stop button (§8.8, §8.9).

**Verify:** force each state — block a query with an injection string, ask something
unanswerable to trigger `unverified`, repeat a query to hit the cache.

---

## 13. Definition of done

- [ ] No raw hex outside the token block in `index.css`
- [ ] Both themes render correctly in all three states: `data-theme="light"`, `data-theme="dark"`, and unstamped system default
- [ ] Answer column measures 60–75 characters at 1920px
- [ ] No horizontal body scroll from 375px to 2560px
- [ ] No `backdrop-filter`, no glow shadow, no gradient fill, no `transition: all`
- [ ] Every animation passes the §9.1 test
- [ ] `prefers-reduced-motion` honoured in both CSS and JS timers
- [ ] Every interactive element has a visible `:focus-visible` state
- [ ] All ten states in §10 are individually reachable and visually distinct
- [ ] Contrast table in §3 re-measured against any changed token
- [ ] Theme toggle persists across reload
- [ ] `npm run build` passes with no TypeScript errors

---

## Appendix — references

The three directions this specification was selected from, with the products they were
extracted from:

| Direction | Source | Outcome |
|---|---|---|
| Enterprise Console | Glean — `#343CED` electric blue, oatmeal neutrals, PolySans, tokenised system rebuilt 2026 on Base UI | **Adopted for the light theme** |
| Answer First | Perplexity — `#FCFCF9` warm paper, `#27251E` warm ink, `#016A71` teal, FK Grotesk | Rejected — optimises for a single-screen product |
| Warm Workspace | Claude — `#FAF9F5` cream, `#D97757` clay, `#262624` warm charcoal dark | **Adopted for the dark theme** |

Motion timings follow the convergent guidance in Material, Atlassian and Photon motion
foundations: 80–150ms for repeated micro-interactions, 150–200ms for enter/exit, 250–300ms
for expand and panel transitions.
