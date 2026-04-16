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
pip install chromadb sentence-transformers rank-bm25 tiktoken anthropic numpy requests beautifulsoup4 pymupdf
```

### 2. Prepare your corpus

The pipeline reads a JSONL file where each line is a JSON object with at least `title`, `slug`, `source`, and `content` fields:

```jsonl
{"title": "Chapter 1", "slug": "chapter-1", "source": "my-book", "date": "2024-01-01", "content": "Full text of the chapter...", "word_count": 450}
{"title": "Blog Post Title", "slug": "blog-post-title", "source": "website", "date": "2024-03-15", "content": "Full text...", "word_count": 320}
```

Helper scripts are included for web scraping and PDF extraction (see [Ingest scripts](#ingest-scripts) below), or you can generate the JSONL from any source.

### 3. Configure paths

Edit the constants at the top of `rag_pipeline.py`:

```python
CORPUS_PATH = "path/to/your/discourses.jsonl"
DB_PATH = "path/to/your/chroma_db"
COLLECTION_NAME = "your_collection"
```

### 4. Build the index

```bash
python rag_pipeline.py build
```

This embeds all chunks with `all-mpnet-base-v2` and stores them in ChromaDB + BM25. Takes ~30 seconds for 500 chunks.

### 5. Search

```bash
# Human-readable output
python rag_pipeline.py search "What does the author say about discipline?"

# JSON for programmatic use
python rag_pipeline.py search "meditation" --json --top-k 8

# Fast mode (skip cross-encoder reranking)
python rag_pipeline.py search "truth" --no-rerank
```

## Using with CLI LLM agents

The `search` command needs no API key — it runs entirely locally. This makes it a perfect tool for any CLI-based LLM agent:

**Claude Code / opencode**: Just ask your question and tell the agent to search the pipeline:

```
> What does the author think about self-improvement? Search the knowledge base.
```

The agent runs `python rag_pipeline.py search "self-improvement"`, reads the retrieved passages, and answers using them as context.

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

## Ingest scripts

Two helper scripts are included for building a corpus from web pages and PDFs. These were built for a specific use case but can be adapted for any content.

### `scrape_discourses.py` — Web scraper

Scrapes articles from a paginated WordPress category page. Outputs individual markdown files with YAML frontmatter + a compiled JSONL and HTML book.

```bash
python scrape_discourses.py --dry-run        # Discover URLs only
python scrape_discourses.py                  # Full scrape
python scrape_discourses.py --limit 5        # Test with 5 posts
python scrape_discourses.py --force          # Re-scrape existing files
```

### `process_books.py` — PDF book processor

Extracts chapters from PDF books using either embedded TOC metadata or text-based table of contents parsing. Integrates extracted chapters into the unified corpus alongside web-scraped content.

Both scripts output to `output/` with this structure:

```
output/
  posts/                          # Individual markdown files
    2024-01-01_chapter-title.md
    ...
  discourses.jsonl                # All records, one per line (RAG input)
  manifest.json                   # Metadata index
  kapil_gupta_discourses.html     # Compiled readable HTML book
```

## Adapting for your own content

1. **Generate a JSONL file** with your documents. Each line needs `title`, `slug`, `source`, `content` at minimum.
2. **Update `CORPUS_PATH`** and `DB_PATH` in `rag_pipeline.py`.
3. **Update `COLLECTION_NAME`** to something descriptive.
4. **Update `SYSTEM_PROMPT`** in `rag_pipeline.py` if using chat mode — tailor it to your domain.
5. Run `python rag_pipeline.py build`, then `search`.

The scraper and book processor are optional — use them if your sources are web articles or PDFs, otherwise just produce the JSONL directly.

## Requirements

- Python 3.10+
- ~2GB disk for models (downloaded on first run)
- No GPU required (CPU inference is fast enough for search)
- `ANTHROPIC_API_KEY` env var only needed for `chat` mode
