"""Local markdown/recursive splitters (chunking/splitters.py).

These replaced langchain_text_splitters, whose package __init__ eagerly imports a
sentence-transformers splitter and with it PyTorch — 453 MB resident, to split
strings. That was most of the service's memory footprint and put it over the 512 MB
ceiling of every free hosting tier.
"""

from chunking.splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class TestMarkdownHeaderTextSplitter:
    def test_tracks_the_full_heading_path(self):
        md = "# Title\n\nIntro.\n\n## Section 1\n\nBody.\n\n### Sub A\n\nDetail."
        docs = MarkdownHeaderTextSplitter().split_text(md)

        assert len(docs) == 3
        assert docs[0].metadata == {"Header 1": "Title"}
        assert docs[1].metadata == {"Header 1": "Title", "Header 2": "Section 1"}
        assert docs[2].metadata == {
            "Header 1": "Title", "Header 2": "Section 1", "Header 3": "Sub A",
        }

    def test_keeps_heading_lines_in_the_content(self):
        """strip_headers=False: the heading is part of the passage's text."""
        docs = MarkdownHeaderTextSplitter().split_text("## Alpha\n\nBody text.")
        assert "## Alpha" in docs[0].page_content

    def test_a_sibling_heading_clears_the_deeper_level(self):
        """Otherwise a section is attributed to a subsection it is not under."""
        md = "## A\n\n### A1\n\nx\n\n## B\n\ny"
        docs = MarkdownHeaderTextSplitter().split_text(md)

        last = docs[-1]
        assert last.metadata.get("Header 2") == "B"
        assert "Header 3" not in last.metadata, "stale subsection leaked into section B"

    def test_hashes_inside_code_fences_are_not_headings(self):
        """A shell comment would otherwise shred the document."""
        md = "## Setup\n\n```bash\n# install deps\npip install x\n# run it\n```\n\nDone."
        docs = MarkdownHeaderTextSplitter().split_text(md)

        assert len(docs) == 1
        assert "pip install x" in docs[0].page_content

    def test_text_with_no_headings_returns_one_section(self):
        docs = MarkdownHeaderTextSplitter().split_text("Just prose, no headings.")
        assert len(docs) == 1
        assert docs[0].metadata == {}


class TestRecursiveCharacterTextSplitter:
    def test_short_text_is_not_split(self):
        s = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        assert s.split_text("short") == ["short"]

    def test_long_text_respects_the_size_budget(self):
        s = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = s.split_text("This is a sentence. " * 60)

        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 100 + 20  # separator reattachment slack

    def test_no_content_is_lost(self):
        s = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=0)
        words = [f"w{i}" for i in range(300)]
        chunks = s.split_text(" ".join(words))

        combined = " ".join(chunks)
        for w in (words[0], words[150], words[-1]):
            assert w in combined

    def test_overlap_carries_context_between_chunks(self):
        s = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=40)
        chunks = s.split_text(" ".join(f"word{i}" for i in range(200)))

        assert len(chunks) > 1
        # The tail of one chunk should reappear at the head of the next.
        assert any(
            set(a.split()) & set(b.split())
            for a, b in zip(chunks, chunks[1:])
        )

    def test_an_unbroken_run_is_hard_cut_rather_than_hanging(self):
        """A base64 blob has no separator to split on; it must still terminate."""
        s = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
        chunks = s.split_text("A" * 500)

        assert len(chunks) > 1
        assert all(len(c) <= 60 for c in chunks)

    def test_overlap_must_be_smaller_than_chunk_size(self):
        """Otherwise merging cannot make progress and would loop."""
        import pytest

        with pytest.raises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
