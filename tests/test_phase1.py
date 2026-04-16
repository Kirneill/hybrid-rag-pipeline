"""Phase 1 tests: broken workflow fixes."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_class_renamed():
    """HybridRAG exists, KapilGuptaRAG does not."""
    import rag_pipeline
    assert hasattr(rag_pipeline, "HybridRAG"), "HybridRAG class not found"
    assert not hasattr(rag_pipeline, "KapilGuptaRAG"), "KapilGuptaRAG should not exist"
    print("  PASS: class renamed to HybridRAG")


def test_corpus_path_default():
    """Default CORPUS_PATH ends with corpus.jsonl, not discourses.jsonl."""
    from rag_pipeline import CORPUS_PATH
    assert CORPUS_PATH.endswith("corpus.jsonl"), f"Expected corpus.jsonl, got {CORPUS_PATH}"
    assert "discourses" not in os.path.basename(CORPUS_PATH)
    print("  PASS: CORPUS_PATH defaults to corpus.jsonl")


def test_config_override():
    """Config file overrides module constants."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, "rag_config.json")

    had_config = os.path.isfile(config_path)
    try:
        with open(config_path, "w") as f:
            json.dump({"collection_name": "test_override", "max_chunk_tokens": 999}, f)

        # Force reimport
        if "rag_pipeline" in sys.modules:
            del sys.modules["rag_pipeline"]
        import rag_pipeline

        assert rag_pipeline.COLLECTION_NAME == "test_override", \
            f"Expected test_override, got {rag_pipeline.COLLECTION_NAME}"
        assert rag_pipeline.MAX_CHUNK_TOKENS == 999, \
            f"Expected 999, got {rag_pipeline.MAX_CHUNK_TOKENS}"
        print("  PASS: config file overrides work")
    finally:
        if not had_config:
            os.remove(config_path)
        else:
            pass
        if "rag_pipeline" in sys.modules:
            del sys.modules["rag_pipeline"]


def test_config_missing_is_fine():
    """Missing config file should not error."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, "rag_config.json")

    had_config = os.path.isfile(config_path)
    backup = None
    if had_config:
        with open(config_path) as f:
            backup = f.read()
        os.remove(config_path)

    try:
        if "rag_pipeline" in sys.modules:
            del sys.modules["rag_pipeline"]
        import rag_pipeline
        assert rag_pipeline.COLLECTION_NAME == "default"
        print("  PASS: missing config file handled gracefully")
    finally:
        if backup is not None:
            with open(config_path, "w") as f:
                f.write(backup)
        if "rag_pipeline" in sys.modules:
            del sys.modules["rag_pipeline"]


def test_requirements_txt_exists():
    """requirements.txt exists and has correct packages."""
    req_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.txt")
    assert os.path.isfile(req_path), "requirements.txt not found"
    with open(req_path) as f:
        content = f.read()
    required = ["chromadb", "sentence-transformers", "rank-bm25", "tiktoken", "anthropic"]
    for pkg in required:
        assert pkg in content, f"Missing {pkg} in requirements.txt"
    print("  PASS: requirements.txt exists with all packages")


def test_example_config_exists():
    """rag_config.example.json exists and is valid JSON."""
    example_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_config.example.json"
    )
    assert os.path.isfile(example_path), "rag_config.example.json not found"
    with open(example_path) as f:
        data = json.load(f)
    assert "corpus_path" in data
    assert "collection_name" in data
    print("  PASS: rag_config.example.json valid")


def test_no_kapil_in_pipeline_code():
    """No Kapil-specific strings in rag_pipeline.py."""
    pipeline_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_pipeline.py"
    )
    with open(pipeline_path) as f:
        code = f.read()

    forbidden = ["KapilGupta", "Kapil Gupta", "kapil_gupta", "kapilgupta"]
    for term in forbidden:
        assert term not in code, f"Found '{term}' in rag_pipeline.py — should be generic"
    print("  PASS: no Kapil-specific strings in pipeline")


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_pipeline_loads_existing_index():
    """HybridRAG can load the existing ChromaDB index."""
    if "rag_pipeline" in sys.modules:
        del sys.modules["rag_pipeline"]
    from rag_pipeline import HybridRAG, DB_PATH

    if not os.path.isdir(DB_PATH):
        print("  SKIP: no existing index to test against")
        return

    rag = HybridRAG()
    rag.load_index()
    assert len(rag.chunks) > 0, "No chunks loaded"
    assert rag.bm25 is not None, "BM25 not built"
    assert rag.chroma_collection is not None, "ChromaDB not loaded"
    print(f"  PASS: loaded {len(rag.chunks)} chunks from existing index")
    return rag


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------

def test_search_e2e(rag=None):
    """Full search pipeline: query → hybrid search → results."""
    if rag is None:
        if "rag_pipeline" in sys.modules:
            del sys.modules["rag_pipeline"]
        from rag_pipeline import HybridRAG, DB_PATH
        if not os.path.isdir(DB_PATH):
            print("  SKIP: no index for E2E test")
            return
        rag = HybridRAG()
        rag.load_index()

    results = rag.search("What is truth?", top_k=3, use_rerank=False)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all("title" in r and "content" in r and "source" in r for r in results)
    assert all(r["rank"] == i for i, r in enumerate(results, 1))
    print(f"  PASS: E2E search returned {len(results)} results with correct structure")


def test_search_json_output():
    """CLI search --json produces valid JSON on stdout."""
    import subprocess
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag_pipeline.py")
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")

    if not os.path.isdir(db_path):
        print("  SKIP: no index for CLI test")
        return

    result = subprocess.run(
        [sys.executable, script, "search", "truth", "--json", "--top-k", "2", "--no-rerank"],
        capture_output=True, timeout=120,
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr.decode('utf-8', errors='replace')}"
    data = json.loads(result.stdout.decode("utf-8", errors="replace"))
    assert len(data) == 2
    assert data[0]["rank"] == 1
    print("  PASS: CLI --json output is valid JSON")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1 TESTS")
    print("=" * 60)

    print("\n--- Unit Tests ---")
    test_class_renamed()
    test_corpus_path_default()
    test_config_override()
    test_config_missing_is_fine()
    test_requirements_txt_exists()
    test_example_config_exists()
    test_no_kapil_in_pipeline_code()

    print("\n--- Integration Tests ---")
    rag = test_pipeline_loads_existing_index()

    print("\n--- E2E Tests ---")
    test_search_e2e(rag)
    test_search_json_output()

    print("\n" + "=" * 60)
    print("ALL PHASE 1 TESTS PASSED")
    print("=" * 60)
