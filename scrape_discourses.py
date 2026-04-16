"""
Scraper for Kapil Gupta's Discourses.
https://www.kapilguptamd.com/category/discourses/

Outputs:
  - Individual markdown files with YAML frontmatter
  - discourses.jsonl (one JSON object per line)
  - kapil_gupta_discourses.html (self-contained HTML book)
  - manifest.json

Dependencies: requests, beautifulsoup4
"""

import argparse
import datetime
import html
import json
import os
import re
import time
import urllib3

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.kapilguptamd.com"
CATEGORY_URL = BASE_URL + "/category/discourses/"
POST_URL_PATTERN = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/([\w-]+)/?$")

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

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
.subtitle { color: #666; font-size: 1rem; margin-bottom: 2rem; }
.date { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
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


def create_session():
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)
    return session


def fetch_page(session, url, verify_ssl=True):
    """Fetch a URL. On SSL error, retry once with verify=False."""
    try:
        resp = session.get(url, timeout=30, verify=verify_ssl)
        return resp
    except requests.exceptions.SSLError:
        if verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            print(f"  SSL error for {url}, retrying without verification")
            return fetch_page(session, url, verify_ssl=False)
        raise
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        print(f"  WARNING: Connection error for {url}: {exc}")
        return None


def discover_urls(session, delay):
    """Paginate through the discourses category and collect all post URLs."""
    all_urls = []
    seen = set()
    page_num = 1

    while True:
        if page_num == 1:
            url = CATEGORY_URL
        else:
            url = CATEGORY_URL + f"page/{page_num}/"

        print(f"Discovering page {page_num}: {url}")
        resp = fetch_page(session, url)

        if resp is None:
            print(f"  Failed to fetch page {page_num}, stopping discovery")
            break

        if resp.status_code == 404:
            print(f"  Page {page_num} returned 404, end of pagination")
            break

        if resp.status_code != 200:
            print(f"  WARNING: Page {page_num} returned HTTP {resp.status_code}, stopping")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=True)
        page_found = 0

        for link in links:
            href = link["href"]
            match = POST_URL_PATTERN.search(href)
            if match:
                # Normalize: ensure it starts with BASE_URL
                if href.startswith("/"):
                    href = BASE_URL + href
                if href not in seen:
                    seen.add(href)
                    all_urls.append(href)
                    page_found += 1

        print(f"  Found {page_found} new URLs (total: {len(all_urls)})")
        page_num += 1
        time.sleep(delay)

    return all_urls


def parse_url_parts(url):
    """Extract date and slug from a discourse URL."""
    match = POST_URL_PATTERN.search(url)
    if not match:
        return None, None
    year, month, day, slug = match.groups()
    date_str = f"{year}-{month}-{day}"
    return date_str, slug


BIO_MARKER = "Kapil Gupta is a personal advisor to"
BOOK_TITLES = {
    "Atmamun",
    "A Master\u2019s Secret Whispers",
    "A Master's Secret Whispers",
    "Direct Truth",
}


def _is_boilerplate(text):
    """Check if a paragraph is part of the recurring bio/book-list footer."""
    if text.startswith(BIO_MARKER):
        return True
    if text == "His books include:":
        return True
    if text.startswith("Related Reading:"):
        return True
    for title in BOOK_TITLES:
        if text.startswith(title):
            return True
    return False


def parse_post(html_text):
    """Parse a discourse post page, returning (title, content_paragraphs, image_url)."""
    soup = BeautifulSoup(html_text, "html.parser")

    # Title: h1 inside article, fallback to first h1
    title = None
    article = soup.find("article")
    if article:
        h1 = article.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    if not title:
        title = "Untitled"

    # Content: div.post-content -> all <p> tags
    paragraphs = []
    content_div = soup.find("div", class_="post-content")
    if content_div:
        for p in content_div.find_all("p"):
            text = p.get_text(strip=True)
            if text and not _is_boilerplate(text):
                paragraphs.append(text)

    # Featured image: figure.post-image -> img src
    image_url = None
    figure = soup.find("figure", class_="post-image")
    if figure:
        img = figure.find("img")
        if img and img.get("src"):
            image_url = img["src"]

    return title, paragraphs, image_url


def escape_frontmatter_title(title):
    """Escape a title for YAML frontmatter double-quoted string."""
    escaped = title.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_markdown(title, slug, url, date_str, paragraphs, image_url):
    """Build the markdown file content with YAML frontmatter."""
    word_count = sum(len(p.split()) for p in paragraphs)
    content = "\n\n".join(paragraphs)

    lines = [
        "---",
        f"title: {escape_frontmatter_title(title)}",
        f"slug: {slug}",
        f"url: {url}",
        f"date: {date_str}",
        "author: Kapil Gupta",
        f"word_count: {word_count}",
        f"image_url: {image_url if image_url else ''}",
        "---",
        "",
        f"# {title}",
        "",
        content,
        "",
    ]
    return "\n".join(lines)


def md_filename(date_str, slug):
    return f"{date_str}_{slug}.md"


def parse_frontmatter(file_content):
    """Parse frontmatter from a markdown file. Returns (metadata_dict, body_text)."""
    # Split on --- markers
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
        # Strip surrounding quotes
        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        metadata[key] = value

    return metadata, body


def load_all_posts_from_disk(posts_dir):
    """Read all .md files from the posts directory and return parsed records sorted by date."""
    records = []
    if not os.path.isdir(posts_dir):
        return records

    for fname in os.listdir(posts_dir):
        if not fname.endswith(".md"):
            continue
        filepath = os.path.join(posts_dir, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()

        metadata, body = parse_frontmatter(file_content)
        if not metadata:
            continue

        # Strip the "# Title" line from body to get just paragraphs
        body_lines = body.split("\n")
        content_lines = []
        for line in body_lines:
            if line.startswith("# "):
                continue
            content_lines.append(line)
        content_text = "\n".join(content_lines).strip()

        date_str = metadata.get("date", "")
        word_count_str = metadata.get("word_count", "0")
        try:
            word_count = int(word_count_str)
        except ValueError:
            word_count = 0

        image_url = metadata.get("image_url", "")
        if not image_url:
            image_url = None

        records.append({
            "title": metadata.get("title", "Untitled"),
            "slug": metadata.get("slug", ""),
            "url": metadata.get("url", ""),
            "date": date_str,
            "year": int(date_str[:4]) if len(date_str) >= 4 else 0,
            "content": content_text,
            "word_count": word_count,
            "image_url": image_url,
            "file": fname,
        })

    records.sort(key=lambda r: r["date"])
    return records


def write_jsonl(output_dir, records):
    """Write discourses.jsonl from records."""
    scraped_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    jsonl_path = os.path.join(output_dir, "discourses.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            obj = {
                "title": rec["title"],
                "slug": rec["slug"],
                "url": rec["url"],
                "date": rec["date"],
                "year": rec["year"],
                "content": rec["content"],
                "word_count": rec["word_count"],
                "image_url": rec["image_url"],
                "scraped_at": scraped_at,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {jsonl_path} ({len(records)} records)")


def write_manifest(output_dir, records):
    """Write manifest.json."""
    scraped_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dates = [r["date"] for r in records if r["date"]]
    earliest = min(dates) if dates else ""
    latest = max(dates) if dates else ""

    manifest = {
        "source": CATEGORY_URL,
        "author": "Kapil Gupta",
        "total_posts": len(records),
        "date_range": {"earliest": earliest, "latest": latest},
        "scraped_at": scraped_at,
        "posts": [
            {
                "title": r["title"],
                "slug": r["slug"],
                "date": r["date"],
                "file": r["file"],
                "word_count": r["word_count"],
            }
            for r in records
        ],
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote {manifest_path}")


def write_html_book(output_dir, records):
    """Generate a self-contained HTML book from all records."""
    if not records:
        print("No records to generate HTML book from.")
        return

    dates = [r["date"] for r in records if r["date"]]
    earliest = min(dates) if dates else "?"
    latest = max(dates) if dates else "?"

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append("<title>Kapil Gupta — Discourses</title>")
    parts.append(f"<style>\n{HTML_CSS}</style>")
    parts.append("</head>")
    parts.append("<body>")

    # Title page
    parts.append("<h1>Kapil Gupta &mdash; Discourses</h1>")
    parts.append(
        f'<div class="subtitle">{len(records)} discourses &middot; '
        f'{html.escape(earliest)} to {html.escape(latest)}</div>'
    )

    # Table of contents
    parts.append('<div class="toc">')
    parts.append("<h2>Table of Contents</h2>")
    parts.append("<ol>")
    for rec in records:
        anchor = f"disc-{rec['date']}-{rec['slug']}"
        escaped_title = html.escape(rec["title"])
        escaped_date = html.escape(rec["date"])
        parts.append(
            f'<li><a href="#{anchor}">{escaped_title}</a>'
            f'<span class="toc-date">{escaped_date}</span></li>'
        )
    parts.append("</ol>")
    parts.append("</div>")

    # Discourse sections
    for rec in records:
        anchor = f"disc-{rec['date']}-{rec['slug']}"
        escaped_title = html.escape(rec["title"])
        escaped_date = html.escape(rec["date"])
        parts.append(f'<div class="discourse" id="{anchor}">')
        parts.append(f"<h2>{escaped_title}</h2>")
        parts.append(f'<div class="date">{escaped_date}</div>')

        # Split content into paragraphs
        content_paragraphs = rec["content"].split("\n\n")
        for para in content_paragraphs:
            para = para.strip()
            if para:
                parts.append(f"<p>{html.escape(para)}</p>")

        parts.append("</div>")

    parts.append("</body>")
    parts.append("</html>")

    html_path = os.path.join(output_dir, "kapil_gupta_discourses.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Wrote {html_path}")


def existing_slugs(posts_dir):
    """Return set of (date, slug) tuples for existing markdown files."""
    slugs = set()
    if not os.path.isdir(posts_dir):
        return slugs
    for fname in os.listdir(posts_dir):
        if fname.endswith(".md"):
            # filename pattern: YYYY-MM-DD_slug.md
            stem = fname[:-3]  # remove .md
            parts = stem.split("_", 1)
            if len(parts) == 2:
                slugs.add((parts[0], parts[1]))
    return slugs


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Kapil Gupta's Discourses from kapilguptamd.com"
    )
    parser.add_argument(
        "--output", "-o",
        default="./output",
        help="Output directory path (default: ./output)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between requests (default: 2.0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of posts to fetch (default: no limit)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape even if markdown file already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover URLs and print count, do not fetch individual posts",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    posts_dir = os.path.join(output_dir, "posts")
    os.makedirs(posts_dir, exist_ok=True)

    session = create_session()

    # Step 1: Discover URLs
    print("=== Discovering discourse URLs ===")
    urls = discover_urls(session, args.delay)
    print(f"\nDiscovered {len(urls)} discourse URLs")

    if args.dry_run:
        print("\n--dry-run specified. URLs found:")
        for i, url in enumerate(urls, 1):
            print(f"  {i}. {url}")
        print(f"\nTotal: {len(urls)}")
        return

    # Apply limit
    if args.limit is not None:
        urls = urls[:args.limit]
        print(f"Limited to {len(urls)} posts")

    # Step 2: Scrape individual posts
    print(f"\n=== Scraping {len(urls)} posts ===")
    existing = existing_slugs(posts_dir)
    scraped = 0
    skipped = 0
    failed = 0

    for i, url in enumerate(urls, 1):
        date_str, slug = parse_url_parts(url)
        if date_str is None:
            print(f"[{i}/{len(urls)}] WARNING: Could not parse URL {url}, skipping")
            failed += 1
            continue

        # Idempotency check
        if not args.force and (date_str, slug) in existing:
            skipped += 1
            continue

        resp = fetch_page(session, url)
        if resp is None:
            print(f"[{i}/{len(urls)}] FAILED: {url}")
            failed += 1
            time.sleep(args.delay)
            continue

        if resp.status_code == 404:
            print(f"[{i}/{len(urls)}] WARNING: 404 for {url}, skipping")
            failed += 1
            time.sleep(args.delay)
            continue

        if resp.status_code != 200:
            print(f"[{i}/{len(urls)}] WARNING: HTTP {resp.status_code} for {url}, skipping")
            failed += 1
            time.sleep(args.delay)
            continue

        title, paragraphs, image_url = parse_post(resp.text)

        if not paragraphs:
            print(f"[{i}/{len(urls)}] WARNING: Empty content for {url}, skipping")
            failed += 1
            time.sleep(args.delay)
            continue

        md_content = build_markdown(title, slug, url, date_str, paragraphs, image_url)
        filepath = os.path.join(posts_dir, md_filename(date_str, slug))
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        scraped += 1
        print(f"[{i}/{len(urls)}] {title}")
        time.sleep(args.delay)

    print(f"\n=== Scraping complete ===")
    print(f"  Scraped: {scraped}")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  Failed: {failed}")

    # Step 3: Rebuild aggregate files from all .md files on disk
    print("\n=== Rebuilding aggregate files ===")
    records = load_all_posts_from_disk(posts_dir)
    print(f"Loaded {len(records)} posts from disk")

    write_jsonl(output_dir, records)
    write_manifest(output_dir, records)
    write_html_book(output_dir, records)

    print("\nDone.")


if __name__ == "__main__":
    main()
