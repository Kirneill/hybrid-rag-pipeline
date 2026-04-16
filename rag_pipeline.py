"""
Hybrid RAG pipeline for Q&A over any document corpus.

Architecture:
    Query → Query Expansion (LLM) → Hybrid Search (Vector + BM25)
    → Reciprocal Rank Fusion → Cross-Encoder Reranking → LLM Answer

Usage:
    python rag_pipeline.py build                    # Build the index (once)
    python rag_pipeline.py search "query"           # Search-only (no API key needed)
    python rag_pipeline.py search "query" --json    # JSON output
    python rag_pipeline.py chat                     # Interactive Q&A with LLM
"""

import argparse
import io
import json
import os
import pickle
import re
import sys
import textwrap
import time
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import anthropic
import chromadb
import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "output", "corpus.jsonl")
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "default"

EMBED_MODEL_NAME = "all-mpnet-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "claude-sonnet-4-6"

MAX_CHUNK_TOKENS = 1500
SPLIT_CHUNK_TOKENS = 800
OVERLAP_TOKENS = 100


def _load_config():
    """Load optional config from rag_config.json alongside this script."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_config.json")
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


_config = _load_config()
CORPUS_PATH = _config.get("corpus_path", CORPUS_PATH)
DB_PATH = _config.get("db_path", DB_PATH)
COLLECTION_NAME = _config.get("collection_name", COLLECTION_NAME)
EMBED_MODEL_NAME = _config.get("embed_model", EMBED_MODEL_NAME)
RERANK_MODEL_NAME = _config.get("rerank_model", RERANK_MODEL_NAME)
LLM_MODEL = _config.get("llm_model", LLM_MODEL)
MAX_CHUNK_TOKENS = _config.get("max_chunk_tokens", MAX_CHUNK_TOKENS)
SPLIT_CHUNK_TOKENS = _config.get("split_chunk_tokens", SPLIT_CHUNK_TOKENS)
OVERLAP_TOKENS = _config.get("overlap_tokens", OVERLAP_TOKENS)

QUERY_EXPANSION_PROMPT = textwrap.dedent("""\
    Given this question about a collection of writings, generate exactly 2 \
    alternative phrasings that might match relevant passages. Return ONLY \
    the 2 alternatives, one per line, no numbering or bullets.

    Question: {query}""")

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a knowledgeable assistant answering questions based on a curated \
    collection of writings. You have been given relevant passages retrieved \
    from the corpus.

    Rules:
    - Answer based ONLY on the provided context passages. Do not use prior knowledge.
    - Quote specific phrases from the passages when they directly answer the question.
    - If the context doesn't contain enough information to fully answer, say so explicitly.
    - Cite which source each point comes from (e.g., title and source name).
    - Keep answers concise but thorough. Aim for 2-4 paragraphs.""")

# ANSI color helpers
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Custom ChromaDB embedding function wrapping sentence-transformers
# ---------------------------------------------------------------------------

class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
    """Wraps a SentenceTransformer model so ChromaDB uses the same embedder
    for both indexing and querying."""

    def __init__(self, model: SentenceTransformer):
        self._model = model

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        embeddings = self._model.encode(input, convert_to_numpy=True)
        return [row for row in embeddings]


# ---------------------------------------------------------------------------
# Tokenizer utility
# ---------------------------------------------------------------------------

_enc = None


def count_tokens(text: str) -> int:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# Main RAG class
# ---------------------------------------------------------------------------

class HybridRAG:
    def __init__(
        self,
        corpus_path: str = CORPUS_PATH,
        db_path: str = DB_PATH,
        top_k: int = 5,
    ):
        self.corpus_path = corpus_path
        self.db_path = db_path
        self.top_k = top_k

        self.chunks: list[dict] = []
        self._id_to_chunk: dict[str, dict] = {}
        self.embed_model: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        self.chroma_collection = None
        self.bm25: Optional[BM25Okapi] = None
        self.llm_client: Optional[anthropic.Anthropic] = None

        self.use_expansion = True
        self.use_rerank = True

    # ------------------------------------------------------------------
    # Corpus loading & chunking
    # ------------------------------------------------------------------

    def load_corpus(self) -> list[dict]:
        """Load JSONL and chunk large documents. Returns list of chunk dicts."""
        with open(self.corpus_path, encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]

        chunks: list[dict] = []
        for idx, doc in enumerate(docs):
            base_id = f"{doc['source']}_{idx:03d}_{doc['slug']}"
            token_count = count_tokens(doc["content"])

            if token_count <= MAX_CHUNK_TOKENS:
                chunks.append({
                    "id": base_id,
                    "title": doc["title"],
                    "source": doc["source"],
                    "date": doc.get("date", ""),
                    "content": doc["content"],
                    "token_count": token_count,
                })
            else:
                sub_chunks = self._split_document(doc["content"])
                for ci, sub_text in enumerate(sub_chunks):
                    chunks.append({
                        "id": f"{base_id}_chunk_{ci}",
                        "title": doc["title"],
                        "source": doc["source"],
                        "date": doc.get("date", ""),
                        "content": sub_text,
                        "token_count": count_tokens(sub_text),
                    })

        self.chunks = chunks
        self._id_to_chunk = {c["id"]: c for c in self.chunks}
        return chunks

    def _split_document(self, text: str) -> list[str]:
        """Split a long document into ~800-token chunks with 100-token overlap,
        splitting at paragraph boundaries (double newline). Falls back to
        sentence-level splitting for text without paragraph breaks."""
        paragraphs = re.split(r"\n\n+", text)

        # If splitting produced a single block that's still too large,
        # break it into sentences instead.
        expanded: list[str] = []
        for para in paragraphs:
            if count_tokens(para) > SPLIT_CHUNK_TOKENS:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                expanded.extend(sentences)
            else:
                expanded.append(para)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for part in expanded:
            part_tokens = count_tokens(part)

            if current_tokens + part_tokens > SPLIT_CHUNK_TOKENS and current_parts:
                # Emit current chunk
                chunks.append(" ".join(current_parts))

                # Build overlap: walk backwards until we have ~OVERLAP_TOKENS
                overlap_parts: list[str] = []
                overlap_tokens = 0
                for p in reversed(current_parts):
                    pt = count_tokens(p)
                    if overlap_tokens + pt > OVERLAP_TOKENS and overlap_parts:
                        break
                    overlap_parts.insert(0, p)
                    overlap_tokens += pt

                current_parts = overlap_parts + [part]
                current_tokens = overlap_tokens + part_tokens
            else:
                current_parts.append(part)
                current_tokens += part_tokens

        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self):
        """Embed all chunks and store in ChromaDB. Also builds BM25 index."""
        self._init_embed_model()

        print(f"Loading corpus from {self.corpus_path}...", file=sys.stderr)
        chunks = self.load_corpus()
        print(f"  {len(chunks)} chunks from {self._count_unique_docs()} documents", file=sys.stderr)

        # Create/reset ChromaDB collection
        client = chromadb.PersistentClient(path=self.db_path)
        embed_fn = SentenceTransformerEmbeddingFunction(self.embed_model)

        # Delete existing collection if it exists, then create fresh
        try:
            client.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Embed and add in batches
        batch_size = 64
        total = len(chunks)
        print(f"Embedding and indexing {total} chunks...", file=sys.stderr)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = chunks[start:end]

            ids = [c["id"] for c in batch]
            documents = [c["content"] for c in batch]
            metadatas = [
                {
                    "title": c["title"],
                    "source": c["source"],
                    "date": c["date"],
                    "token_count": c["token_count"],
                }
                for c in batch
            ]

            # Compute embeddings explicitly
            embeddings = self.embed_model.encode(documents, convert_to_numpy=True)
            embedding_list = [row.tolist() for row in embeddings]

            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_list,
            )
            print(f"  Indexed {end}/{total} chunks", file=sys.stderr)

        self.chroma_collection = collection
        self._build_bm25()
        self._save_bm25_cache()
        print(f"Index built successfully at {self.db_path}", file=sys.stderr)

    def _count_unique_docs(self) -> int:
        """Count unique document titles in chunks."""
        return len({c["title"] for c in self.chunks})

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def load_index(self):
        """Load existing ChromaDB index and rebuild BM25 from stored chunks."""
        self._init_embed_model()

        client = chromadb.PersistentClient(path=self.db_path)
        embed_fn = SentenceTransformerEmbeddingFunction(self.embed_model)

        try:
            self.chroma_collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn,
            )
        except Exception:
            print(f"No index found at {self.db_path}. Run 'build' subcommand first.", file=sys.stderr)
            raise SystemExit(1)

        # Try loading BM25 + chunks from pickle cache (fast path)
        if self._load_bm25_cache():
            self._id_to_chunk = {c["id"]: c for c in self.chunks}
            return

        # Fallback: retrieve all chunks from ChromaDB to rebuild BM25 and local chunk list
        count = self.chroma_collection.count()
        result = self.chroma_collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        self.chunks = []
        for i in range(len(result["ids"])):
            meta = result["metadatas"][i]
            self.chunks.append({
                "id": result["ids"][i],
                "title": meta["title"],
                "source": meta["source"],
                "date": meta.get("date", ""),
                "content": result["documents"][i],
                "token_count": meta.get("token_count", 0),
            })

        self._build_bm25()
        self._id_to_chunk = {c["id"]: c for c in self.chunks}

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def _build_bm25(self):
        """Build BM25 index from self.chunks."""
        tokenized = [c["content"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _save_bm25_cache(self):
        """Persist chunks and BM25 tokenized corpus to pickle for fast loading."""
        cache_path = os.path.join(self.db_path, "bm25_cache.pkl")
        data = {
            "chunks": self.chunks,
            "tokenized": [c["content"].lower().split() for c in self.chunks],
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def _load_bm25_cache(self) -> bool:
        """Load chunks and BM25 from pickle cache. Returns True on success."""
        cache_path = os.path.join(self.db_path, "bm25_cache.pkl")
        if not os.path.isfile(cache_path):
            return False
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.bm25 = BM25Okapi(data["tokenized"])
        return True

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _init_embed_model(self):
        if self.embed_model is None:
            print(f"Loading embedding model ({EMBED_MODEL_NAME})...", file=sys.stderr)
            self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    def _init_reranker(self):
        if self.reranker is None:
            print(f"Loading reranking model ({RERANK_MODEL_NAME})...", file=sys.stderr)
            self.reranker = CrossEncoder(RERANK_MODEL_NAME)

    def _init_llm_client(self):
        if self.llm_client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
                raise SystemExit(1)
            self.llm_client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> list[str]:
        """Use LLM to generate 2 alternate query phrasings."""
        self._init_llm_client()
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        response = self.llm_client.messages.create(
            model=LLM_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return lines[:2]

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

    def vector_search(self, query: str, n: int = 20) -> list[str]:
        """Search ChromaDB, return ranked list of chunk IDs."""
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        results = self.chroma_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(n, self.chroma_collection.count()),
        )
        return results["ids"][0]

    def bm25_search(self, query: str, n: int = 20) -> list[str]:
        """Search BM25 index, return ranked list of chunk IDs."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:n]
        return [self.chunks[i]["id"] for i in top_indices]

    def hybrid_search(self, queries: list[str], n: int = 20) -> list[str]:
        """Run vector + BM25 for each query, merge with RRF."""
        ranked_lists = []
        for q in queries:
            ranked_lists.append(self.vector_search(q, n=n))
            ranked_lists.append(self.bm25_search(q, n=n))
        fused = reciprocal_rank_fusion(ranked_lists)
        return fused[:n]

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(self, query: str, chunk_ids: list[str], n: int = 5) -> list[tuple[str, float]]:
        """Cross-encoder reranking. Returns (chunk_id, score) pairs."""
        self._init_reranker()

        pairs = []
        valid_ids = []
        for cid in chunk_ids:
            if cid in self._id_to_chunk:
                pairs.append((query, self._id_to_chunk[cid]["content"]))
                valid_ids.append(cid)

        scores = self.reranker.predict(pairs)
        scored = list(zip(valid_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    # ------------------------------------------------------------------
    # Search-only pipeline (no LLM needed)
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: Optional[int] = None,
               use_rerank: Optional[bool] = None) -> list[dict]:
        """Hybrid search + optional reranking. Returns structured results.

        No LLM API key required. Each result dict contains:
            rank, title, source, date, relevance_score, content
        """
        if top_k is None:
            top_k = self.top_k
        if use_rerank is None:
            use_rerank = self.use_rerank

        # Hybrid search (vector + BM25 → RRF)
        candidate_ids = self.hybrid_search([query], n=20)

        # Reranking
        if use_rerank:
            scored = self.rerank(query, candidate_ids, n=top_k)
        else:
            scored = [(cid, 0.0) for cid in candidate_ids[:top_k]]

        results = []
        for rank, (cid, score) in enumerate(scored, 1):
            chunk = self._id_to_chunk[cid]
            results.append({
                "rank": rank,
                "title": chunk["title"],
                "source": chunk["source"],
                "date": chunk.get("date", ""),
                "relevance_score": float(score),
                "content": chunk["content"],
            })

        return results

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def generate_answer(self, query: str, chunks: list[dict]) -> str:
        """Feed context chunks to LLM, return answer."""
        self._init_llm_client()
        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] Title: {c['title']} | Source: {c['source']}\n{c['content']}"
            )
        context = "\n\n".join(context_parts)

        response = self.llm_client.messages.create(
            model=LLM_MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """Full pipeline. Returns {"answer": str, "sources": list, "timing": dict}."""
        timing = {}

        # Step 1: Query expansion
        t0 = time.time()
        if self.use_expansion:
            alt_queries = self.expand_query(question)
            all_queries = [question] + alt_queries
        else:
            all_queries = [question]
        timing["expansion"] = time.time() - t0

        n_queries = len(all_queries)
        print(
            f"{_DIM}[Searching... {n_queries} quer{'ies' if n_queries > 1 else 'y'} "
            f"x 2 indexes → 20 candidates"
            f"{'→ reranking ' if self.use_rerank else ''}"
            f"→ top {self.top_k}]{_RESET}"
        )

        # Step 2: Hybrid search
        t0 = time.time()
        candidate_ids = self.hybrid_search(all_queries, n=20)
        timing["search"] = time.time() - t0

        # Step 3: Reranking
        t0 = time.time()
        if self.use_rerank:
            scored = self.rerank(question, candidate_ids, n=self.top_k)
        else:
            scored = [(cid, 0.0) for cid in candidate_ids[: self.top_k]]
        timing["rerank"] = time.time() - t0

        top_chunks = []
        sources = []
        for cid, score in scored:
            chunk = self._id_to_chunk[cid]
            top_chunks.append(chunk)
            sources.append({
                "title": chunk["title"],
                "source": chunk["source"],
                "score": float(score),
            })

        # Step 4: Answer generation
        t0 = time.time()
        answer = self.generate_answer(question, top_chunks)
        timing["generation"] = time.time() - t0

        timing["total"] = sum(timing.values())

        return {
            "answer": answer,
            "sources": sources,
            "top_chunks": top_chunks,
            "timing": timing,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def ensure_index_loaded(self):
        """Load index if not already loaded. Used by CLI entry points."""
        if self.chroma_collection is None:
            self.load_index()


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Merge multiple ranked lists using RRF.

    For each document, score = sum(1 / (k + rank)) across all lists where it appears.
    """
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_search_results(results: list[dict]) -> str:
    """Format search results for human/agent-readable stdout output."""
    lines = [f"=== Search Results ({len(results)} chunks) ===\n"]
    separator = "-" * 48
    for r in results:
        date_str = f", {r['date']}" if r["date"] else ""
        lines.append(
            f"[{r['rank']}] \"{r['title']}\" ({r['source']}{date_str})"
            f" -- relevance: {r['relevance_score']:.2f}"
        )
        lines.append(separator)
        lines.append(r["content"])
        lines.append(separator)
        lines.append("")
    return "\n".join(lines)


def _cmd_build(args):
    """Handler for the 'build' subcommand."""
    rag = HybridRAG()
    rag.build_index()


def _cmd_search(args):
    """Handler for the 'search' subcommand."""
    rag = HybridRAG(top_k=args.top_k)
    rag.use_rerank = not args.no_rerank
    rag.load_index()
    results = rag.search(args.query, top_k=args.top_k, use_rerank=not args.no_rerank)

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(_format_search_results(results))


def _cmd_chat(args):
    """Handler for the 'chat' subcommand (interactive Q&A with LLM)."""
    rag = HybridRAG(top_k=args.top_k)
    rag.use_expansion = not args.no_expansion
    rag.use_rerank = not args.no_rerank

    rag.load_index()
    n_docs = rag._count_unique_docs()
    n_chunks = len(rag.chunks)

    print(f"\n{_BOLD}Knowledge Base{_RESET} "
          f"({n_docs} documents, {n_chunks} chunks)")
    print("Type your question, or 'quit' to exit.\n")

    try:
        while True:
            try:
                question = input(f"{_GREEN}> {_RESET}").strip()
            except EOFError:
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            print()
            result = rag.query(question)

            # Show retrieved chunks if requested
            if args.show_chunks:
                print(f"\n{_BOLD}Retrieved Chunks:{_RESET}")
                for i, chunk in enumerate(result["top_chunks"], 1):
                    print(f"\n{_CYAN}[{i}] {chunk['title']}{_RESET} "
                          f"{_DIM}({chunk['source']}){_RESET}")
                    print(textwrap.indent(chunk["content"][:500], "    "))
                    if len(chunk["content"]) > 500:
                        print(f"    {_DIM}... ({chunk['token_count']} tokens total){_RESET}")
                print()

            # Answer
            print(f"\n{_BOLD}Answer:{_RESET}")
            print(result["answer"])

            # Sources
            print(f"\n{_BOLD}Sources:{_RESET}")
            for i, src in enumerate(result["sources"], 1):
                score_str = f"{src['score']:.2f} relevance" if src["score"] > 0 else ""
                print(f"  {_DIM}[{i}] \"{src['title']}\" ({src['source']})"
                      f"{' — ' + score_str if score_str else ''}{_RESET}")

            # Timing
            t = result["timing"]
            parts = []
            if rag.use_expansion:
                parts.append(f"expansion {t['expansion']:.1f}s")
            parts.append(f"search {t['search']:.1f}s")
            if rag.use_rerank:
                parts.append(f"rerank {t['rerank']:.1f}s")
            parts.append(f"generation {t['generation']:.1f}s")
            print(f"\n{_DIM}[{' | '.join(parts)} | total {t['total']:.1f}s]{_RESET}\n")

    except KeyboardInterrupt:
        print(f"\n{_DIM}Interrupted. Goodbye.{_RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG Pipeline — search any document corpus with vector + BM25"
    )
    # Backward compat: --build still works as a top-level flag
    parser.add_argument(
        "--build", action="store_true", help=argparse.SUPPRESS,
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- build ---
    sub_build = subparsers.add_parser(
        "build", help="Build/rebuild the index from JSONL"
    )

    # --- search ---
    sub_search = subparsers.add_parser(
        "search",
        help="Search the corpus (no LLM API key needed)",
    )
    sub_search.add_argument(
        "query", type=str, help="Search query string"
    )
    sub_search.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)",
    )
    sub_search.add_argument(
        "--no-rerank", action="store_true",
        help="Disable cross-encoder reranking (faster)",
    )
    sub_search.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )

    # --- chat ---
    sub_chat = subparsers.add_parser(
        "chat", help="Interactive Q&A with LLM (requires ANTHROPIC_API_KEY)"
    )
    sub_chat.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks for context (default: 5)",
    )
    sub_chat.add_argument(
        "--no-expansion", action="store_true",
        help="Disable query expansion",
    )
    sub_chat.add_argument(
        "--no-rerank", action="store_true",
        help="Disable cross-encoder reranking",
    )
    sub_chat.add_argument(
        "--show-chunks", action="store_true",
        help="Print retrieved chunks before answer",
    )

    args = parser.parse_args()

    # Backward compat: --build flag
    if args.build:
        _cmd_build(args)
        return

    if args.command == "build":
        _cmd_build(args)
    elif args.command == "search":
        _cmd_search(args)
    elif args.command == "chat":
        _cmd_chat(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
