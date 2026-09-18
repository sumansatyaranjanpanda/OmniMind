"""Markdown-header and recursive-character splitters.

These replace the two `langchain_text_splitters` splitters this project used.
That package's `__init__.py` eagerly imports every splitter it ships, including
`sentence_transformers`, which pulls in transformers and PyTorch — measured at
**453 MB of resident memory, imported purely to split strings**. Importing the
submodule directly doesn't avoid it either: Python executes the parent package's
`__init__` first, so there is no escape short of not depending on it.

That single import was most of the application's memory footprint (idle RSS 721 MB,
of which `chunking.strategies` alone accounted for 413 MB) and put the service over
the 512 MB ceiling of every free hosting tier. Removing it is also just correct:
nothing here needs a neural network to find a line starting with "##".

Both implementations match the behaviour the previous ones were used with —
`strip_headers=False` for the markdown splitter, and largest-separator-first
recursion with overlap for the character splitter — and are covered by
tests/unit/test_chunking.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Only h1-h3 are treated as structural. Deeper headings are left inline as content,
# matching how this was configured before: splitting on h4+ tends to isolate single
# sentences that carry no independent meaning.
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$")


@dataclass
class SplitDocument:
    """One split section: its text, plus the heading hierarchy above it."""

    page_content: str
    metadata: dict[str, str] = field(default_factory=dict)


class MarkdownHeaderTextSplitter:
    """Split markdown into sections at h1-h3 boundaries.

    Each returned section carries the full heading path that was open at that point
    (``{"Header 1": ..., "Header 2": ...}``), so a passage can still say which part
    of which document it came from after being separated from its context.
    """

    def __init__(self, headers_to_split_on: list[tuple[str, str]] | None = None,
                 strip_headers: bool = False) -> None:
        # Accepted for signature compatibility with how this was previously called.
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"),
        ]
        self.strip_headers = strip_headers
        self._level_names = {
            len(hashes): name for hashes, name in self.headers_to_split_on
        }

    def split_text(self, text: str) -> list[SplitDocument]:
        docs: list[SplitDocument] = []
        # Heading path currently in scope, by level: {1: "Title", 2: "Section 1"}.
        active: dict[int, str] = {}
        buffer: list[str] = []

        def flush() -> None:
            if not buffer:
                return
            content = "\n".join(buffer).strip()
            if content:
                docs.append(
                    SplitDocument(
                        page_content=content,
                        metadata={
                            self._level_names[lvl]: title
                            for lvl, title in sorted(active.items())
                            if lvl in self._level_names
                        },
                    )
                )
            buffer.clear()

        in_code_fence = False
        for line in text.splitlines():
            # A "#" inside a fenced code block is a comment, not a heading. Without
            # this, any shell or Python snippet silently shreds the document.
            if line.lstrip().startswith("```"):
                in_code_fence = not in_code_fence
                buffer.append(line)
                continue

            match = None if in_code_fence else _HEADER_RE.match(line)
            if match and len(match.group(1)) in self._level_names:
                # The preceding section ends here, before the heading is recorded.
                flush()

                level = len(match.group(1))
                active[level] = match.group(2).strip()
                # A new h2 invalidates the h3 beneath the previous h2; leaving it
                # would attribute this section to a subsection it isn't under.
                for deeper in [lvl for lvl in active if lvl > level]:
                    del active[deeper]

                if not self.strip_headers:
                    buffer.append(line)
            else:
                buffer.append(line)

        flush()
        return docs


class RecursiveCharacterTextSplitter:
    """Split text to a size budget, preferring the most natural break available.

    Tries separators in order — paragraph, then line, then sentence, then word —
    and only falls back to cutting mid-word when a single run has no break in it
    at all. Splitting on the largest structure that fits keeps a chunk from
    starting or ending mid-thought.
    """

    def __init__(self, chunk_size: int = 1800, chunk_overlap: int = 200,
                 separators: list[str] | None = None,
                 length_function=len) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.length_function = length_function

    def split_text(self, text: str) -> list[str]:
        if self.length_function(text) <= self.chunk_size:
            return [text] if text.strip() else []
        return self._merge(self._explode(text, self.separators))

    def _explode(self, text: str, separators: list[str]) -> list[str]:
        """Break text into the largest fragments that still fit the budget."""
        if self.length_function(text) <= self.chunk_size:
            return [text]

        if not separators:
            # No separators left: hard-cut. Only reachable for a single
            # unbroken run longer than the budget, e.g. a base64 blob.
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep, rest = separators[0], separators[1:]
        if sep == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        pieces: list[str] = []
        for part in text.split(sep):
            if not part:
                continue
            # Put the separator back so the text reads correctly once rejoined.
            candidate = part + sep
            if self.length_function(candidate) <= self.chunk_size:
                pieces.append(candidate)
            else:
                pieces.extend(self._explode(candidate, rest))
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        """Recombine fragments up to the budget, carrying overlap between chunks."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = self.length_function(piece)
            if current and current_len + piece_len > self.chunk_size:
                joined = "".join(current).strip()
                if joined:
                    chunks.append(joined)

                # Carry the tail of this chunk into the next one, so a fact sitting
                # on a boundary appears whole in at least one of them.
                carry: list[str] = []
                carry_len = 0
                for prev in reversed(current):
                    prev_len = self.length_function(prev)
                    if carry_len + prev_len > self.chunk_overlap:
                        break
                    carry.insert(0, prev)
                    carry_len += prev_len
                current = carry
                current_len = carry_len

            current.append(piece)
            current_len += piece_len

        if current:
            joined = "".join(current).strip()
            if joined:
                chunks.append(joined)

        return chunks
