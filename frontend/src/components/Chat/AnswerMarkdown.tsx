import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { visit } from 'unist-util-visit';
import type { Root, Element, Text } from 'hast';
import { CitationItem } from '../../types';

/**
 * Renders an answer as real markdown, with [^N] citation markers kept clickable.
 *
 * The previous renderer only split on newlines, so every `###` and `**bold**` the model
 * emitted reached the user as literal punctuation. Handing the text to a markdown parser
 * fixes that, but introduces a conflict worth explaining: `[^1]` *is* GFM footnote syntax.
 * remark-gfm claims those markers while building the syntax tree, so by the time we could
 * inspect the output they have already been rewritten into footnote links pointing at a
 * footnotes section that does not exist.
 *
 * So the markers are swapped for inert private-use sentinels before parsing — text the
 * parser has no rules for and passes through untouched — and a small rehype plugin turns
 * them back into citation elements afterwards. Doing the substitution in the tree rather
 * than on the rendered output means citations work identically inside paragraphs, list
 * items, table cells and headings, with no per-element special casing.
 */

const OPEN = '';
const CLOSE = '';

// Tolerates the model grouping numbers into one bracket ("[^3, ^4]") — each number
// becomes its own chip rather than the whole bracket failing to match a citation.
const MARKER_RE = /\[\^[\d,\s^]+\]/g;
const SENTINEL_RE = /(\d+)/;

function protectMarkers(content: string): string {
  return content.replace(MARKER_RE, (bracket) => {
    const numbers = bracket.match(/\d+/g) || [];
    return numbers.map((n) => `${OPEN}${n}${CLOSE}`).join('');
  });
}

/** Splits sentinel-bearing text nodes into `<citation>` elements. */
function rehypeCitations() {
  return (tree: Root) => {
    visit(tree, 'text', (node: Text, index, parent) => {
      if (!parent || index === null || index === undefined) return;
      if (!node.value.includes(OPEN)) return;

      const parts: Array<Text | Element> = [];
      let rest = node.value;

      for (;;) {
        const match = SENTINEL_RE.exec(rest);
        if (!match || match.index === undefined) break;
        if (match.index > 0) {
          parts.push({ type: 'text', value: rest.slice(0, match.index) });
        }
        parts.push({
          type: 'element',
          tagName: 'citation',
          properties: { marker: `[^${match[1]}]` },
          children: [],
        });
        rest = rest.slice(match.index + match[0].length);
      }

      if (rest) parts.push({ type: 'text', value: rest });
      parent.children.splice(index, 1, ...parts);
      return index + parts.length;
    });
  };
}

interface AnswerMarkdownProps {
  content: string;
  citations?: CitationItem[];
  onCitationClick: (citation: CitationItem) => void;
}

export const AnswerMarkdown: React.FC<AnswerMarkdownProps> = ({
  content,
  citations,
  onCitationClick,
}) => {
  const prepared = React.useMemo(() => protectMarkers(content), [content]);

  const components = React.useMemo(
    () => ({
      citation: ({ marker }: { marker?: string }) => {
        const found = citations?.find((c) => c.marker === marker);

        // No matching citation — render it inert rather than clickable. This happens on
        // every cache hit: the server replays the stored answer text, markers and all,
        // but returns `citations: []` because finalize_cached never restores the list.
        // Inventing a placeholder here would put fabricated provenance in front of the
        // user, which is the one thing a citation-first product cannot do.
        if (!found) {
          return (
            <span
              className="citation-ref citation-ref-inert"
              title="Source details aren't available for this answer"
            >
              {marker}
            </span>
          );
        }

        return (
          <button
            className="citation-ref"
            onClick={() => onCitationClick(found)}
            title={`View evidence source for ${marker}`}
          >
            {marker}
          </button>
        );
      },
      // Links from web citations open in a new tab; noreferrer because the target is
      // untrusted content pulled from a search engine.
      a: ({ href, children }: { href?: string; children?: React.ReactNode }) => (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      ),
      // Wide tables scroll inside their own box instead of stretching the message column.
      table: ({ children }: { children?: React.ReactNode }) => (
        <div className="md-table-wrap">
          <table>{children}</table>
        </div>
      ),
    }),
    [citations, onCitationClick]
  );

  return (
    <div className="md-answer">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeCitations]}
        components={components as never}
      >
        {prepared}
      </ReactMarkdown>
    </div>
  );
};
