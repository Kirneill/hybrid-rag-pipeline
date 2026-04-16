"""
Process Kapil Gupta's PDF books and transcript into the discourse corpus.

Extracts chapters from 4 PDF books, processes a transcript, deduplicates
against existing website discourses, and rebuilds unified corpus files.

Dependencies: pymupdf (import fitz), stdlib only otherwise.
"""

import datetime
import html
import json
import os
import re

import fitz

PDF_DIR = "G:/My Drive/PDF/Gupta/"
OUTPUT_DIR = "F:/CLAUDE/kapil-gupta-discourses/output/"
POSTS_DIR = os.path.join(OUTPUT_DIR, "posts")

# Book prefixes for filename matching
BOOK_PREFIXES = {
    "masters_secret_whispers": "A Master",
    "atmamun": "Atmamun",
    "direct_truth": "Direct Truth",
    "complete_collection": "Kapil Gupta MD",
}

# Chapters to skip (not content)
SKIP_TITLES_MASTERS = {"About The Author"}
SKIP_TITLES_ATMAMUN = {"About the Author", "Siddha Performance", "Websites and Media"}
SKIP_TITLES_DIRECT_TRUTH = {"ABOUT THE AUTHOR"}

HTML_CSS = """\
body {
    font-family: Georgia, 'Times New Roman', serif;
    max-width: 65ch;
    margin: 2rem auto;
    padding: 0 1rem;
    line-height: 1.7;
    color: #1a1a1a;
    background: #fafaf8;
}
h1 { font-size: 2.2rem; margin-bottom: 0.3rem; }
h2 { font-size: 1.6rem; margin-top: 2rem; margin-bottom: 0.2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
h3 { font-size: 1.3rem; margin-top: 1.5rem; color: #444; }
.subtitle { color: #666; font-size: 1rem; margin-bottom: 2rem; }
.date { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
.source-label { color: #666; font-size: 0.85rem; font-style: italic; }
.toc { margin: 2rem 0; }
.toc ol { padding-left: 1.5rem; }
.toc li { margin-bottom: 0.3rem; }
.toc a { text-decoration: none; color: #2a5db0; }
.toc a:hover { text-decoration: underline; }
.toc .toc-date { color: #999; font-size: 0.85rem; margin-left: 0.5rem; }
.discourse { margin-top: 3rem; }
p { margin-bottom: 1rem; }
@media print {
    .discourse { page-break-before: always; }
    .toc { page-break-after: always; }
    body { max-width: none; margin: 0; padding: 1cm; }
}
"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def slugify(text, max_len=60):
    """Convert text to a URL-friendly slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s


def normalize_title(title):
    """Normalize a title for dedup comparison."""
    s = title.lower().strip()
    # Strip trailing periods and ellipses
    s = re.sub(r"[.\u2026]+$", "", s)
    # Strip punctuation except apostrophes inside words
    # First, protect internal apostrophes
    s = re.sub(r"(?<=\w)['\u2019](?=\w)", "\x00", s)
    # Remove all other punctuation
    s = re.sub(r"[^\w\s\x00]", "", s)
    # Restore apostrophes
    s = s.replace("\x00", "'")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def escape_frontmatter_title(title):
    """Escape a title for YAML frontmatter double-quoted string."""
    escaped = title.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def clean_page_text(text):
    """Clean extracted PDF page text."""
    text = text.strip()
    # Remove page numbers: lines that are just a number (with optional whitespace)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped and re.match(r"^\d+$", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def extract_paragraphs(text):
    """Split text into paragraphs, stripping each one."""
    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")
    paragraphs = re.split(r"\n\s*\n", text)
    result = []
    for p in paragraphs:
        p = p.strip()
        if p:
            # Collapse internal whitespace within paragraph lines
            p = re.sub(r"[ \t]+", " ", p)
            result.append(p)
    return result


def build_book_markdown(title, slug, source, paragraphs):
    """Build markdown file content for a book chapter."""
    word_count = sum(len(p.split()) for p in paragraphs)
    content = "\n\n".join(paragraphs)

    lines = [
        "---",
        f"title: {escape_frontmatter_title(title)}",
        f"slug: {slug}",
        "url: ",
        "date: ",
        "author: Kapil Gupta",
        f"source: {source}",
        f"word_count: {word_count}",
        "image_url: ",
        "---",
        "",
        f"# {title}",
        "",
        content,
        "",
    ]
    return "\n".join(lines)


def book_md_filename(source_slug, chapter_num, title_slug):
    """Generate filename for a book chapter."""
    return f"{source_slug}_{chapter_num:03d}_{title_slug}.md"


def find_pdf(prefix):
    """Find a PDF file in PDF_DIR matching the given prefix."""
    for fname in os.listdir(PDF_DIR):
        if fname.startswith(prefix) and fname.endswith(".pdf"):
            return os.path.join(PDF_DIR, fname)
    return None


def parse_frontmatter(file_content):
    """Parse frontmatter from a markdown file. Returns (metadata_dict, body_text)."""
    parts = file_content.split("---", 2)
    if len(parts) < 3:
        return {}, file_content

    frontmatter_text = parts[1].strip()
    body = parts[2].strip()

    metadata = {}
    for line in frontmatter_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        colon_idx = line.find(":")
        if colon_idx == -1:
            continue
        key = line[:colon_idx].strip()
        value = line[colon_idx + 1:].strip()
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        metadata[key] = value

    return metadata, body


# ---------------------------------------------------------------------------
# PDF chapter extraction
# ---------------------------------------------------------------------------

def extract_chapters_with_toc(doc, skip_titles=None):
    """Extract chapters from a PDF that has an embedded TOC."""
    toc = doc.get_toc()
    if skip_titles is None:
        skip_titles = set()

    chapters = []
    for i, (level, title, page_num) in enumerate(toc):
        if title in skip_titles:
            continue

        start_page = page_num - 1  # Convert 1-indexed to 0-indexed

        # End page: next TOC entry's page - 1, or end of document
        if i + 1 < len(toc):
            end_page = toc[i + 1][2] - 2  # next entry's page (1-indexed) - 1 for exclusive, -1 for 0-indexed
        else:
            end_page = len(doc) - 1

        # Extract text from all pages in range
        page_texts = []
        for p in range(start_page, end_page + 1):
            if 0 <= p < len(doc):
                page_texts.append(clean_page_text(doc[p].get_text()))

        full_text = "\n\n".join(t for t in page_texts if t)

        # Strip the chapter title from the beginning of the text
        # The title appears as the first line(s)
        text_lines = full_text.split("\n", 1)
        if text_lines:
            first_line = text_lines[0].strip()
            # Compare case-insensitively since Direct Truth uses ALL CAPS in TOC and text
            if normalize_title(first_line) == normalize_title(title):
                full_text = text_lines[1].strip() if len(text_lines) > 1 else ""
            else:
                # Title might be split across multiple lines (e.g., long Direct Truth titles)
                # Try matching first few lines joined
                rejoined = full_text.lstrip()
                title_normalized = normalize_title(title)
                for num_lines in range(2, 5):
                    candidate_lines = rejoined.split("\n", num_lines)
                    candidate = " ".join(l.strip() for l in candidate_lines[:num_lines])
                    if normalize_title(candidate) == title_normalized:
                        remainder = candidate_lines[num_lines] if len(candidate_lines) > num_lines else ""
                        full_text = remainder.strip()
                        break

        paragraphs = extract_paragraphs(full_text)
        if paragraphs:
            chapters.append((title, paragraphs))

    return chapters


def extract_chapters_text_toc(doc, toc_page_indices, skip_titles=None, content_start_index=None):
    """Extract chapters from a PDF using text-based TOC pages."""
    if skip_titles is None:
        skip_titles = set()

    # Read TOC pages to get chapter titles
    toc_text = ""
    for idx in toc_page_indices:
        toc_text += doc[idx].get_text() + "\n"

    # Parse titles from TOC text (one per line, skip header lines)
    raw_titles = []
    for line in toc_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("contents", "table of contents"):
            continue
        if line in skip_titles:
            continue
        raw_titles.append(line)

    if not raw_titles:
        return []

    # Scan pages to find chapter boundaries
    # For each page, check if its first non-empty line matches a known title
    chapter_boundaries = []  # (title, start_page_index)

    search_start = content_start_index if content_start_index is not None else 0
    title_set = set(raw_titles)
    # Also build normalized versions for fuzzy matching
    norm_to_title = {}
    for t in raw_titles:
        norm_to_title[normalize_title(t)] = t

    for page_idx in range(search_start, len(doc)):
        page_text = doc[page_idx].get_text().strip()
        if not page_text:
            continue

        # Get first non-empty line
        first_line = ""
        for line in page_text.split("\n"):
            line = line.strip()
            if line:
                first_line = line
                break

        if not first_line:
            continue

        # Check exact match first
        if first_line in title_set:
            chapter_boundaries.append((first_line, page_idx))
        else:
            # Try normalized match
            norm_first = normalize_title(first_line)
            if norm_first in norm_to_title:
                chapter_boundaries.append((norm_to_title[norm_first], page_idx))

    # Extract text between consecutive boundaries
    chapters = []
    for i, (title, start_idx) in enumerate(chapter_boundaries):
        if title in skip_titles:
            continue

        if i + 1 < len(chapter_boundaries):
            end_idx = chapter_boundaries[i + 1][1] - 1
        else:
            end_idx = len(doc) - 1

        page_texts = []
        for p in range(start_idx, end_idx + 1):
            page_texts.append(clean_page_text(doc[p].get_text()))

        full_text = "\n\n".join(t for t in page_texts if t)

        # Strip title from beginning
        text_lines = full_text.split("\n", 1)
        if text_lines and text_lines[0].strip() == title:
            full_text = text_lines[1].strip() if len(text_lines) > 1 else ""

        paragraphs = extract_paragraphs(full_text)
        if paragraphs:
            chapters.append((title, paragraphs))

    return chapters


# ---------------------------------------------------------------------------
# Book-specific extraction
# ---------------------------------------------------------------------------

def process_masters_secret_whispers():
    """Extract chapters from A Master's Secret Whispers."""
    pdf_path = find_pdf(BOOK_PREFIXES["masters_secret_whispers"])
    if not pdf_path:
        print("ERROR: Could not find A Master's Secret Whispers PDF")
        return []

    print(f"Processing: A Master's Secret Whispers")
    print(f"  File: {os.path.basename(pdf_path)}")

    doc = fitz.open(pdf_path)
    chapters = extract_chapters_text_toc(
        doc,
        toc_page_indices=[4, 5],
        skip_titles=SKIP_TITLES_MASTERS,
        content_start_index=6,
    )
    doc.close()

    print(f"  Extracted {len(chapters)} chapters")
    return chapters


def process_atmamun():
    """Extract chapters from Atmamun."""
    pdf_path = find_pdf(BOOK_PREFIXES["atmamun"])
    if not pdf_path:
        print("ERROR: Could not find Atmamun PDF")
        return []

    print(f"Processing: Atmamun")
    print(f"  File: {os.path.basename(pdf_path)}")

    doc = fitz.open(pdf_path)
    chapters = extract_chapters_text_toc(
        doc,
        toc_page_indices=[4],
        skip_titles=SKIP_TITLES_ATMAMUN,
        content_start_index=5,
    )
    doc.close()

    print(f"  Extracted {len(chapters)} chapters")
    return chapters


def process_direct_truth():
    """Extract chapters from Direct Truth."""
    pdf_path = find_pdf(BOOK_PREFIXES["direct_truth"])
    if not pdf_path:
        print("ERROR: Could not find Direct Truth PDF")
        return []

    print(f"Processing: Direct Truth")
    print(f"  File: {os.path.basename(pdf_path)}")

    doc = fitz.open(pdf_path)
    chapters = extract_chapters_with_toc(doc, skip_titles=SKIP_TITLES_DIRECT_TRUTH)
    doc.close()

    print(f"  Extracted {len(chapters)} chapters")
    return chapters


def process_complete_collection():
    """Extract chapters from the Complete Collection."""
    pdf_path = find_pdf(BOOK_PREFIXES["complete_collection"])
    if not pdf_path:
        print("ERROR: Could not find Complete Collection PDF")
        return []

    print(f"Processing: Complete Collection")
    print(f"  File: {os.path.basename(pdf_path)}")

    doc = fitz.open(pdf_path)
    chapters = extract_chapters_with_toc(doc)
    doc.close()

    print(f"  Extracted {len(chapters)} chapters")
    return chapters


def process_transcript():
    """Process the DoYouWantTreatmentOrCure.txt transcript."""
    transcript_path = os.path.join(PDF_DIR, "DoYouWantTreatmentOrCure.txt")
    if not os.path.isfile(transcript_path):
        print("ERROR: Could not find transcript file")
        return []

    print(f"Processing: Transcript")
    print(f"  File: DoYouWantTreatmentOrCure.txt")

    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 7:
        print("  ERROR: Transcript file has fewer than 7 lines")
        return []

    # Line 7 (index 6) is the transcript
    raw = lines[6].strip()

    # Strip "Transcript: " prefix and surrounding quotes
    if raw.startswith('Transcript: "'):
        raw = raw[13:]
        if raw.endswith('"'):
            raw = raw[:-1]
    elif raw.startswith("Transcript: "):
        raw = raw[12:]

    # Remove inline timestamps: (00:02), (1:14:31)
    raw = re.sub(r"\(\d{1,2}:\d{2}:\d{2}\)", "", raw)
    raw = re.sub(r"\(\d{1,2}:\d{2}\)", "", raw)

    # Clean up extra spaces from timestamp removal
    raw = re.sub(r"  +", " ", raw).strip()

    paragraphs = extract_paragraphs(raw)
    if not paragraphs:
        # Transcript is one long blob - split into reasonable paragraphs by sentence groups
        # For now, keep as single paragraph blocks split by periods+space patterns
        paragraphs = [raw]

    title = "Do You Want the Treatment or the Cure"
    print(f"  Processed transcript: {len(paragraphs)} paragraphs, {sum(len(p.split()) for p in paragraphs)} words")
    return [(title, paragraphs)]


# ---------------------------------------------------------------------------
# Save chapters to disk
# ---------------------------------------------------------------------------

def save_chapters(chapters, source, source_slug):
    """Save extracted chapters as markdown files. Returns list of saved filenames."""
    saved = []
    for i, (title, paragraphs) in enumerate(chapters, 1):
        title_slug = slugify(title)
        if not title_slug:
            title_slug = f"chapter-{i}"
        filename = book_md_filename(source_slug, i, title_slug)
        md_content = build_book_markdown(title, title_slug, source, paragraphs)
        filepath = os.path.join(POSTS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
        saved.append(filename)
    return saved


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_website_vs_collection(collection_chapters):
    """Remove website .md files that overlap with Complete Collection chapters.

    Returns (replaced_count, kept_website_count).
    """
    # Build mapping of normalized title -> filepath for website files
    website_files = {}
    for fname in os.listdir(POSTS_DIR):
        if not fname.endswith(".md"):
            continue
        # Website files have date-based names like 2025-04-14_slug.md
        if not re.match(r"\d{4}-\d{2}-\d{2}_", fname):
            continue

        filepath = os.path.join(POSTS_DIR, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        metadata, _ = parse_frontmatter(content)
        title = metadata.get("title", "")
        source = metadata.get("source", "")

        # Only consider website files (no source field, or source: website)
        if source and source != "website":
            continue

        norm = normalize_title(title)
        website_files[norm] = filepath

    # Check each collection chapter against website files
    replaced = 0
    collection_norms = set()
    for title, _ in collection_chapters:
        norm = normalize_title(title)
        collection_norms.add(norm)
        if norm in website_files:
            old_path = website_files[norm]
            os.remove(old_path)
            replaced += 1

    kept = len(website_files) - replaced
    return replaced, kept


# ---------------------------------------------------------------------------
# Add source: website to remaining website files
# ---------------------------------------------------------------------------

def add_source_to_website_files():
    """Add source: website to existing website .md files that lack a source field."""
    updated = 0
    for fname in os.listdir(POSTS_DIR):
        if not fname.endswith(".md"):
            continue
        # Website files have date-based names
        if not re.match(r"\d{4}-\d{2}-\d{2}_", fname):
            continue

        filepath = os.path.join(POSTS_DIR, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        metadata, _ = parse_frontmatter(content)
        if "source" in metadata:
            continue

        # Insert source: website after the author: line
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if line.startswith("author:"):
                new_lines.append("source: website")
        new_content = "\n".join(new_lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        updated += 1

    return updated


# ---------------------------------------------------------------------------
# Rebuild unified corpus
# ---------------------------------------------------------------------------

def load_all_posts():
    """Load all .md files from posts directory and return parsed records."""
    records = []
    for fname in sorted(os.listdir(POSTS_DIR)):
        if not fname.endswith(".md"):
            continue
        filepath = os.path.join(POSTS_DIR, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()

        metadata, body = parse_frontmatter(file_content)
        if not metadata:
            continue

        # Strip "# Title" line from body
        body_lines = body.split("\n")
        content_lines = [line for line in body_lines if not line.startswith("# ")]
        content_text = "\n".join(content_lines).strip()

        date_str = metadata.get("date", "")
        try:
            word_count = int(metadata.get("word_count", "0"))
        except ValueError:
            word_count = 0

        image_url = metadata.get("image_url", "")
        if not image_url:
            image_url = None

        source = metadata.get("source", "website")

        year = 0
        if len(date_str) >= 4:
            try:
                year = int(date_str[:4])
            except ValueError:
                pass

        records.append({
            "title": metadata.get("title", "Untitled"),
            "slug": metadata.get("slug", ""),
            "url": metadata.get("url", ""),
            "date": date_str,
            "year": year,
            "source": source,
            "content": content_text,
            "word_count": word_count,
            "image_url": image_url,
            "file": fname,
        })

    return records


def write_jsonl(records):
    """Write discourses.jsonl with source field."""
    scraped_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    jsonl_path = os.path.join(OUTPUT_DIR, "discourses.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            obj = {
                "title": rec["title"],
                "slug": rec["slug"],
                "url": rec["url"],
                "date": rec["date"],
                "year": rec["year"],
                "source": rec["source"],
                "content": rec["content"],
                "word_count": rec["word_count"],
                "image_url": rec["image_url"],
                "scraped_at": scraped_at,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {jsonl_path} ({len(records)} records)")


def write_manifest(records):
    """Write manifest.json organized by source."""
    scraped_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Group by source
    by_source = {}
    for rec in records:
        src = rec["source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(rec)

    sources_info = {}
    for src, recs in by_source.items():
        info = {"count": len(recs)}
        if src == "website":
            dates = [r["date"] for r in recs if r["date"]]
            if dates:
                info["date_range"] = {"earliest": min(dates), "latest": max(dates)}
        sources_info[src] = info

    manifest = {
        "source": "Kapil Gupta Complete Works",
        "total_posts": len(records),
        "sources": sources_info,
        "scraped_at": scraped_at,
        "posts": [
            {
                "title": r["title"],
                "slug": r["slug"],
                "date": r["date"],
                "source": r["source"],
                "file": r["file"],
                "word_count": r["word_count"],
            }
            for r in records
        ],
    }

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote {manifest_path}")


def write_html_book(records):
    """Generate a self-contained HTML book organized by source."""
    if not records:
        print("No records to generate HTML book from.")
        return

    # Source display order and labels
    source_order = [
        ("website", "Discourses"),
        ("complete_collection", "Complete Collection"),
        ("masters_secret_whispers", "A Master\u2019s Secret Whispers"),
        ("atmamun", "Atmamun"),
        ("direct_truth", "Direct Truth"),
        ("transcript", "Transcript"),
    ]

    by_source = {}
    for rec in records:
        src = rec["source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(rec)

    # Sort website entries by date
    if "website" in by_source:
        by_source["website"].sort(key=lambda r: r["date"])

    total = len(records)

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append("<title>Kapil Gupta &mdash; Complete Works</title>")
    parts.append(f"<style>\n{HTML_CSS}</style>")
    parts.append("</head>")
    parts.append("<body>")

    # Title page
    parts.append("<h1>Kapil Gupta &mdash; Complete Works</h1>")
    parts.append(f'<div class="subtitle">{total} entries across {len(by_source)} sources</div>')

    # Table of contents organized by source
    parts.append('<div class="toc">')
    parts.append("<h2>Table of Contents</h2>")

    for source_key, source_label in source_order:
        if source_key not in by_source:
            continue
        recs = by_source[source_key]
        escaped_label = html.escape(source_label)
        parts.append(f"<h3>{escaped_label} ({len(recs)})</h3>")
        parts.append("<ol>")
        for rec in recs:
            anchor = f"disc-{rec['source']}-{rec['slug']}"
            escaped_title = html.escape(rec["title"])
            date_part = ""
            if rec["date"]:
                date_part = f'<span class="toc-date">{html.escape(rec["date"])}</span>'
            parts.append(f'<li><a href="#{anchor}">{escaped_title}</a>{date_part}</li>')
        parts.append("</ol>")

    parts.append("</div>")

    # Content sections
    for source_key, source_label in source_order:
        if source_key not in by_source:
            continue
        recs = by_source[source_key]
        escaped_label = html.escape(source_label)

        for rec in recs:
            anchor = f"disc-{rec['source']}-{rec['slug']}"
            escaped_title = html.escape(rec["title"])
            parts.append(f'<div class="discourse" id="{anchor}">')
            parts.append(f"<h2>{escaped_title}</h2>")
            if rec["date"]:
                parts.append(f'<div class="date">{html.escape(rec["date"])}</div>')
            parts.append(f'<div class="source-label">{escaped_label}</div>')

            content_paragraphs = rec["content"].split("\n\n")
            for para in content_paragraphs:
                para = para.strip()
                if para:
                    parts.append(f"<p>{html.escape(para)}</p>")

            parts.append("</div>")

    parts.append("</body>")
    parts.append("</html>")

    html_path = os.path.join(OUTPUT_DIR, "kapil_gupta_discourses.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Wrote {html_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(POSTS_DIR, exist_ok=True)

    # Step 1: Process each book
    print("=" * 60)
    print("STEP 1: Extract chapters from PDF books")
    print("=" * 60)

    masters_chapters = process_masters_secret_whispers()
    atmamun_chapters = process_atmamun()
    direct_truth_chapters = process_direct_truth()
    complete_collection_chapters = process_complete_collection()
    transcript_chapters = process_transcript()

    # Step 2: Deduplicate - Complete Collection replaces matching website files
    print()
    print("=" * 60)
    print("STEP 2: Deduplicate website vs Complete Collection")
    print("=" * 60)

    replaced, kept = deduplicate_website_vs_collection(complete_collection_chapters)
    print(f"  Replaced {replaced} website files with Complete Collection versions")
    print(f"  Kept {kept} website-only files")

    # Step 3: Save book chapters as markdown
    print()
    print("=" * 60)
    print("STEP 3: Save book chapters as markdown")
    print("=" * 60)

    saved_masters = save_chapters(
        masters_chapters, "masters_secret_whispers", "masters-secret-whispers"
    )
    print(f"  Master's Secret Whispers: {len(saved_masters)} files")

    saved_atmamun = save_chapters(atmamun_chapters, "atmamun", "atmamun")
    print(f"  Atmamun: {len(saved_atmamun)} files")

    saved_direct = save_chapters(direct_truth_chapters, "direct_truth", "direct-truth")
    print(f"  Direct Truth: {len(saved_direct)} files")

    saved_collection = save_chapters(
        complete_collection_chapters, "complete_collection", "complete-collection"
    )
    print(f"  Complete Collection: {len(saved_collection)} files")

    saved_transcript = save_chapters(transcript_chapters, "transcript", "transcript")
    print(f"  Transcript: {len(saved_transcript)} files")

    # Step 4: Add source: website to remaining website files
    print()
    print("=" * 60)
    print("STEP 4: Add source field to website files")
    print("=" * 60)

    updated = add_source_to_website_files()
    print(f"  Updated {updated} website files with source: website")

    # Step 5: Rebuild unified corpus
    print()
    print("=" * 60)
    print("STEP 5: Rebuild unified corpus")
    print("=" * 60)

    records = load_all_posts()
    print(f"  Loaded {len(records)} total posts")

    write_jsonl(records)
    write_manifest(records)
    write_html_book(records)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    by_source = {}
    for rec in records:
        src = rec["source"]
        by_source[src] = by_source.get(src, 0) + 1

    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}")
    print(f"  TOTAL: {len(records)}")
    print()
    total_words = sum(r["word_count"] for r in records)
    print(f"  Total words: {total_words:,}")
    print("\nDone.")


if __name__ == "__main__":
    main()
