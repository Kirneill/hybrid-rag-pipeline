# Hybrid RAG Pipeline

A local hybrid search pipeline for doing Q&A over any collection of books, articles, or documents. Designed to be used as a tool by CLI LLM agents (Claude Code, opencode, aider) — no API key needed for search.

## How it works

```
Your documents (PDFs, web scrapes, markdown)
    ↓
Chunking (paragraph-boundary, ~800 tokens with overlap)
    ↓
Dual indexing: ChromaDB vectors + BM25 keyword index
    ↓
Query → Hybrid Search (vector + BM25) → Reciprocal Rank Fusion → Cross-Encoder Reranking
    ↓
Ranked passages returned to stdout → your LLM agent reasons over them
```

**Why hybrid?** Pure vector search misses exact keyword matches. Pure BM25 misses semantic similarity. Running both and fusing with RRF catches what either alone would miss — in testing, ~40% of relevant results came from only one method.

## Quick start

### 1. Install dependencies

```bash
pip install chromadb sentence-transformers rank-bm25 tiktoken anthropic numpy
```

Additional dependencies for the example ingest scripts:

```bash
pip install requests beautifulsoup4 pymupdf
```

### 2. Prepare your corpus

The pipeline reads a JSONL file where each line is a JSON object with at least `title`, `slug`, `source`, and `content` fields:

```jsonl
{"title": "Chapter 1", "slug": "chapter-1", "source": "my-book", "date": "2024-01-01", "content": "Full text of the chapter...", "word_count": 450}
{"title": "Blog Post Title", "slug": "blog-post-title", "source": "website", "date": "2024-03-15", "content": "Full text...", "word_count": 320}
```

Place your JSONL file at `output/corpus.jsonl` (the default path), or configure the path via `rag_config.json`.

Example ingest scripts for web scraping and PDF extraction are in `examples/`.

### 3. Configure paths

Create a `rag_config.json` in the project root (see `rag_config.example.json`):

```json
{
    "corpus_path": "output/corpus.jsonl",
    "collection_name": "my_collection"
}
```

Only include the settings you want to override — all fields are optional.

### 4. Build the index

```bash
python rag_pipeline.py build
```

Embeds all chunks with `all-mpnet-base-v2` and stores them in ChromaDB + BM25. Takes ~30 seconds for 500 chunks.

### 5. Search

```bash
# Human-readable output
python rag_pipeline.py search "What does the author say about discipline?"

# JSON for programmatic use
python rag_pipeline.py search "discipline" --json --top-k 8

# Fast mode (skip cross-encoder reranking)
python rag_pipeline.py search "discipline" --no-rerank
```

## Using with CLI LLM agents

The `search` command needs no API key — it runs entirely locally. This makes it a tool for any CLI-based LLM agent:

**Claude Code / opencode / aider**: Ask your question and the agent runs `search`, reads the retrieved passages, and answers using them as context.

```
> What does the author think about self-improvement? Search the knowledge base.
```

The agent runs `python rag_pipeline.py search "self-improvement"`, reads the passages, and reasons over them.

**JSON mode** (`--json`) outputs structured results that agents can parse programmatically — each result includes `rank`, `title`, `source`, `date`, `relevance_score`, and `content`.

**All status/progress messages go to stderr**, so stdout stays clean for piping.

## CLI reference

```
python rag_pipeline.py build                          # Build/rebuild the index
python rag_pipeline.py search "query"                 # Search (no API key needed)
python rag_pipeline.py search "query" --json          # JSON output
python rag_pipeline.py search "query" --top-k 10      # More results
python rag_pipeline.py search "query" --no-rerank     # Skip reranking (faster)
python rag_pipeline.py chat                           # Interactive Q&A (needs ANTHROPIC_API_KEY)
```

## Architecture

| Component | What it does | Model |
|-----------|-------------|-------|
| Embedding | Converts text chunks to vectors | `all-mpnet-base-v2` (768-dim) |
| Vector search | Semantic similarity via ChromaDB | Same model |
| BM25 search | Keyword matching via `rank-bm25` | No model (statistical) |
| Rank fusion | Merges vector + BM25 results | Reciprocal Rank Fusion (k=60) |
| Reranking | Full query-document attention scoring | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation | LLM reasoning over passages (chat mode only) | Claude Sonnet |

### Chunking strategy

- Documents under 1500 tokens are kept whole
- Larger documents are split at paragraph boundaries (~800 tokens per chunk)
- 100-token overlap between chunks prevents context loss at boundaries
- Falls back to sentence-level splitting for paragraphs without line breaks

## Example ingest scripts

Two example scripts in `examples/` show how to build a corpus from web pages and PDFs. Adapt them for your own content.

### `examples/scrape_discourses.py` — Web scraper

Scrapes articles from a paginated WordPress category page. Outputs individual markdown files with YAML frontmatter + a compiled JSONL and HTML book.

```bash
python examples/scrape_discourses.py --dry-run    # Discover URLs only
python examples/scrape_discourses.py              # Full scrape
python examples/scrape_discourses.py --limit 5    # Test with 5 posts
```

### `examples/process_books.py` — PDF book processor

Extracts chapters from PDF books using either embedded TOC metadata or text-based table of contents parsing. Integrates extracted chapters into a unified corpus.

Both scripts output to `output/`:

```
output/
  posts/                    # Individual markdown files
  corpus.jsonl              # All records, one per line (RAG input)
  manifest.json             # Metadata index
```

## Adapting for your own content

1. **Generate a JSONL file** with your documents. Each line needs `title`, `slug`, `source`, `content` at minimum.
2. Create a `rag_config.json` with your paths (see `rag_config.example.json`).
3. Optionally update `SYSTEM_PROMPT` in `rag_pipeline.py` for your domain.
4. Run `python rag_pipeline.py build`, then `search`.

## Requirements

- Python 3.10+
- ~2GB disk for models (downloaded on first run)
- No GPU required (CPU inference is fast enough for search)
- `ANTHROPIC_API_KEY` env var only needed for `chat` mode
