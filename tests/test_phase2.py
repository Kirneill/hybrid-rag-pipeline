"""Phase 2 tests: performance and correctness."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_lazy_tiktoken():
    """tiktoken encoder is not loaded at import time."""
    if "rag_pipeline" in sys.modules:
        del sys.modules["rag_pipeline"]

    import rag_pipeline
    assert rag_pipeline._enc is None, "_enc should be None before first use"

    tokens = rag_pipeline.count_tokens("hello world")
    assert tokens > 0
    assert rag_pipeline._enc is not None, "_enc should be initialized after first use"
    print("  PASS: tiktoken lazily initialized")


def test_bm25_cache_exists():
    """BM25 cache file exists after build."""
    from rag_pipeline import DB_PATH
    cache_path = os.path.join(DB_PATH, "bm25_cache.pkl")
    assert os.path.isfile(cache_path), f"BM25 cache not found at {cache_path}"

    # Verify it's a valid file with expected size
    size = os.path.getsize(cache_path)
    assert size > 1000, f"Cache file suspiciously small: {size} bytes"
    print(f"  PASS: BM25 cache exists ({size / 1024:.0f} KB)")


def test_id_to_chunk_built_once():
    """_id_to_chunk is populated after load_index()."""
    from rag_pipeline import HybridRAG, DB_PATH
    if not os.path.isdir(DB_PATH):
        print("  SKIP: no index")
        return

    rag = HybridRAG()
    assert len(rag._id_to_chunk) == 0, "Should be empty before load"
    rag.load_index()
    assert len(rag._id_to_chunk) == len(rag.chunks), \
        f"_id_to_chunk ({len(rag._id_to_chunk)}) != chunks ({len(rag.chunks)})"
    print(f"  PASS: _id_to_chunk built with {len(rag._id_to_chunk)} entries")
    return rag


def test_specific_exception_in_build():
    """build_index catches ValueError, not bare Exception."""
    import inspect
    from rag_pipeline import HybridRAG
    source = inspect.getsource(HybridRAG.build_index)
    assert "except ValueError" in source, "Should catch ValueError specifically"
    assert "except Exception" not in source, "Should not use bare except Exception"
    print("  PASS: build_index catches ValueError specifically")


def test_scraper_pagination_guard():
    """Scraper has page_found == 0 break condition."""
    scraper_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "scrape_discourses.py"
    )
    with open(scraper_path) as f:
        code = f.read()
    assert "page_found == 0" in code, "Missing pagination break on empty page"
    print("  PASS: scraper has empty-page break condition")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_cache_load_works():
    """Loading from BM25 cache populates chunks and BM25."""
    from rag_pipeline import HybridRAG, DB_PATH

    if not os.path.isdir(DB_PATH):
        print("  SKIP: no index")
        return

    rag = HybridRAG()
    loaded = rag._load_bm25_cache()
    assert loaded, "Cache load returned False"
    assert len(rag.chunks) > 0, "No chunks loaded from cache"
    assert rag.bm25 is not None, "BM25 not built from cache"
    print(f"  PASS: cache load works ({len(rag.chunks)} chunks)")


def test_search_uses_cached_id_map():
    """search() doesn't rebuild id_to_chunk — uses self._id_to_chunk."""
    import inspect
    from rag_pipeline import HybridRAG
    source = inspect.getsource(HybridRAG.search)
    assert "id_to_chunk = {" not in source, "search() should not rebuild id_to_chunk"
    assert "self._id_to_chunk" in source, "search() should use self._id_to_chunk"
    print("  PASS: search() uses cached _id_to_chunk")


def test_rerank_uses_cached_id_map():
    """rerank() doesn't rebuild id_to_chunk."""
    import inspect
    from rag_pipeline import HybridRAG
    source = inspect.getsource(HybridRAG.rerank)
    assert "id_to_chunk = {" not in source, "rerank() should not rebuild id_to_chunk"
    assert "self._id_to_chunk" in source
    print("  PASS: rerank() uses cached _id_to_chunk")


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

def test_full_search_after_cache_load():
    """Full search pipeline works after loading from BM25 cache."""
    from rag_pipeline import HybridRAG, DB_PATH
    if not os.path.isdir(DB_PATH):
        print("  SKIP: no index")
        return

    rag = HybridRAG()
    rag.load_index()

    results = rag.search("What is the mind?", top_k=5, use_rerank=False)
    assert len(results) == 5
    titles = [r["title"] for r in results]
    assert any("mind" in t.lower() for t in titles), f"No mind-related results: {titles}"
    print(f"  PASS: full search returns relevant results after cache load")


def test_search_quality_preserved():
    """Hybrid search quality unchanged after Phase 2 changes."""
    from rag_pipeline import HybridRAG, DB_PATH
    if not os.path.isdir(DB_PATH):
        print("  SKIP: no index")
        return

    rag = HybridRAG()
    rag.load_index()

    test_cases = [
        ("stoicism", ["stoic"]),
        ("meditation and its limitations", ["meditat"]),
        ("How does the mind control human beings?", ["mind"]),
    ]

    for query, expected_keywords in test_cases:
        results = rag.search(query, top_k=5, use_rerank=False)
        top_titles = " ".join(r["title"].lower() for r in results)
        top_content = " ".join(r["content"][:200].lower() for r in results)
        combined = top_titles + " " + top_content
        found = any(kw in combined for kw in expected_keywords)
        assert found, f"Query '{query}': no relevant results found"

    print(f"  PASS: search quality preserved across {len(test_cases)} test queries")


def test_phase1_still_passes():
    """Phase 1 tests still pass (regression check)."""
    from tests.test_phase1 import (
        test_class_renamed, test_corpus_path_default,
        test_requirements_txt_exists, test_example_config_exists,
        test_no_kapil_in_pipeline_code,
    )
    test_class_renamed()
    test_corpus_path_default()
    test_requirements_txt_exists()
    test_example_config_exists()
    test_no_kapil_in_pipeline_code()
    print("  PASS: Phase 1 regression tests pass")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2 TESTS")
    print("=" * 60)

    print("\n--- Unit Tests ---")
    test_lazy_tiktoken()
    test_bm25_cache_exists()
    rag = test_id_to_chunk_built_once()
    test_specific_exception_in_build()
    test_scraper_pagination_guard()

    print("\n--- Integration Tests ---")
    test_cache_load_works()
    test_search_uses_cached_id_map()
    test_rerank_uses_cached_id_map()

    print("\n--- E2E Tests ---")
    test_full_search_after_cache_load()
    test_search_quality_preserved()
    test_phase1_still_passes()

    print("\n" + "=" * 60)
    print("ALL PHASE 2 TESTS PASSED")
    print("=" * 60)
