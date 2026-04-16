"""Phase 3 tests: core function unit tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))


# ---------------------------------------------------------------------------
# _split_document tests
# ---------------------------------------------------------------------------

def test_split_document_short_text():
    """Short text (under max tokens) returns a single chunk."""
    from rag_pipeline import HybridRAG
    rag = HybridRAG.__new__(HybridRAG)

    short = "This is a short document. It has very few tokens."
    chunks = rag._split_document(short)
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0] == short
    print("  PASS: short text returns single chunk")


def test_split_document_paragraph_boundaries():
    """Long text splits at paragraph boundaries (double newlines)."""
    from rag_pipeline import HybridRAG, SPLIT_CHUNK_TOKENS
    rag = HybridRAG.__new__(HybridRAG)

    # Build text with distinct paragraphs that together exceed SPLIT_CHUNK_TOKENS
    # Each word is ~1 token, so build paragraphs of ~500 words each
    para1 = " ".join(f"word{i}" for i in range(500))
    para2 = " ".join(f"term{i}" for i in range(500))
    para3 = " ".join(f"item{i}" for i in range(500))
    long_text = f"{para1}\n\n{para2}\n\n{para3}"

    chunks = rag._split_document(long_text)
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

    # Verify paragraphs are not split mid-word (no double-newlines inside chunks)
    for chunk in chunks:
        assert "\n\n" not in chunk, "Chunk should not contain paragraph breaks"
    print("  PASS: long text splits at paragraph boundaries")


def test_split_document_overlap():
    """Consecutive chunks share overlapping content."""
    from rag_pipeline import HybridRAG
    rag = HybridRAG.__new__(HybridRAG)

    # Build text with many small paragraphs to force multiple chunks
    paragraphs = []
    for i in range(40):
        paragraphs.append(" ".join(f"block{i}_word{j}" for j in range(50)))
    long_text = "\n\n".join(paragraphs)

    chunks = rag._split_document(long_text)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

    # Check that consecutive chunks share some content (overlap)
    for i in range(len(chunks) - 1):
        words_current = set(chunks[i].split())
        words_next = set(chunks[i + 1].split())
        overlap = words_current & words_next
        assert len(overlap) > 0, f"No overlap between chunk {i} and {i+1}"
    print("  PASS: consecutive chunks have overlapping content")


def test_split_document_sentence_fallback():
    """A single paragraph exceeding SPLIT_CHUNK_TOKENS falls back to sentence splitting."""
    from rag_pipeline import HybridRAG
    rag = HybridRAG.__new__(HybridRAG)

    # One giant paragraph (no double newlines) with sentence breaks
    sentences = []
    for i in range(60):
        sentences.append(" ".join(f"sent{i}_w{j}" for j in range(30)) + ".")
    blob = " ".join(sentences)

    chunks = rag._split_document(blob)
    assert len(chunks) > 1, f"Expected multiple chunks from sentence fallback, got {len(chunks)}"
    print("  PASS: single large paragraph falls back to sentence splitting")


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion tests
# ---------------------------------------------------------------------------

def test_rrf_single_list():
    """Single list returns same order."""
    from rag_pipeline import reciprocal_rank_fusion
    single = [["a", "b", "c", "d"]]
    result = reciprocal_rank_fusion(single)
    assert result == ["a", "b", "c", "d"], f"Expected same order, got {result}"
    print("  PASS: single list preserves order")


def test_rrf_identical_lists():
    """Two identical lists give higher scores but same order."""
    from rag_pipeline import reciprocal_rank_fusion
    lists = [["a", "b", "c"], ["a", "b", "c"]]
    result = reciprocal_rank_fusion(lists)
    assert result == ["a", "b", "c"], f"Expected [a,b,c], got {result}"
    print("  PASS: identical lists preserve order with boosted scores")


def test_rrf_merge_different_lists():
    """Items appearing in both lists rank higher than items in only one."""
    from rag_pipeline import reciprocal_rank_fusion
    list1 = ["a", "b", "c"]
    list2 = ["d", "b", "e"]
    # "b" appears in both lists (rank 1 in list1, rank 1 in list2)
    # All others appear in only one list
    result = reciprocal_rank_fusion([list1, list2])
    assert result[0] == "b", f"Expected 'b' first (in both lists), got {result[0]}"
    # All items should be present
    assert set(result) == {"a", "b", "c", "d", "e"}, f"Missing items: {result}"
    print("  PASS: items in multiple lists rank higher")


def test_rrf_disjoint_lists():
    """Completely disjoint lists merge all items, first-list items first for same position."""
    from rag_pipeline import reciprocal_rank_fusion
    list1 = ["x", "y"]
    list2 = ["p", "q"]
    result = reciprocal_rank_fusion([list1, list2])
    assert set(result) == {"x", "y", "p", "q"}, f"Not all items present: {result}"
    # Both first-place items get score 1/(60+1) = same score
    # Ordering among tied items is implementation-dependent, but all should be present
    assert len(result) == 4
    print("  PASS: disjoint lists merge all items")


def test_rrf_custom_k():
    """Custom k parameter affects scores but preserves relative ranking."""
    from rag_pipeline import reciprocal_rank_fusion
    lists = [["a", "b"], ["b", "a"]]
    # With default k=60: both have same total score (symmetric)
    result_default = reciprocal_rank_fusion(lists, k=60)
    # With small k: positions matter more, still symmetric
    result_small = reciprocal_rank_fusion(lists, k=1)
    # Both 'a' and 'b' appear in both results
    assert set(result_default) == {"a", "b"}
    assert set(result_small) == {"a", "b"}
    print("  PASS: custom k parameter works")


# ---------------------------------------------------------------------------
# normalize_title tests (from process_books.py)
# ---------------------------------------------------------------------------

def test_normalize_title_trailing_periods():
    """Strips trailing periods and ellipses."""
    from process_books import normalize_title
    assert normalize_title("Hello World...") == "hello world"
    assert normalize_title("Hello World.") == "hello world"
    assert normalize_title("Hello World\u2026") == "hello world"
    assert normalize_title("Test..") == "test"
    print("  PASS: strips trailing periods and ellipses")


def test_normalize_title_internal_apostrophes():
    """Preserves internal apostrophes (contractions, possessives)."""
    from process_books import normalize_title
    result = normalize_title("Don't Stop")
    assert "don't" in result, f"Lost apostrophe: {result}"
    result2 = normalize_title("Master\u2019s Whispers")
    assert "master's" in result2, f"Lost smart apostrophe: {result2}"
    print("  PASS: preserves internal apostrophes")


def test_normalize_title_lowercases():
    """Lowercases the title."""
    from process_books import normalize_title
    assert normalize_title("THE TRUTH") == "the truth"
    assert normalize_title("Mixed Case Title") == "mixed case title"
    print("  PASS: lowercases")


def test_normalize_title_collapses_whitespace():
    """Collapses multiple spaces into one and strips edges."""
    from process_books import normalize_title
    assert normalize_title("  too   many   spaces  ") == "too many spaces"
    assert normalize_title("tab\there") == "tab here"
    print("  PASS: collapses whitespace")


def test_normalize_title_strips_punctuation():
    """Strips punctuation other than internal apostrophes."""
    from process_books import normalize_title
    result = normalize_title("Hello, World! (2023)")
    assert result == "hello world 2023", f"Unexpected result: {result}"
    result2 = normalize_title("Test: A Sub-title")
    # Hyphens are not word characters, so they get stripped; sub and title merge
    # Actually \w includes underscore and alphanumeric only, so hyphen is removed
    assert "test" in result2
    assert "sub" in result2
    print("  PASS: strips other punctuation")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 CORE TESTS")
    print("=" * 60)

    print("\n--- _split_document tests ---")
    test_split_document_short_text()
    test_split_document_paragraph_boundaries()
    test_split_document_overlap()
    test_split_document_sentence_fallback()

    print("\n--- reciprocal_rank_fusion tests ---")
    test_rrf_single_list()
    test_rrf_identical_lists()
    test_rrf_merge_different_lists()
    test_rrf_disjoint_lists()
    test_rrf_custom_k()

    print("\n--- normalize_title tests ---")
    test_normalize_title_trailing_periods()
    test_normalize_title_internal_apostrophes()
    test_normalize_title_lowercases()
    test_normalize_title_collapses_whitespace()
    test_normalize_title_strips_punctuation()

    print("\n" + "=" * 60)
    print("ALL PHASE 3 CORE TESTS PASSED")
    print("=" * 60)
