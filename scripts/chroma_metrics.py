"""
Metrics report for the ChromaDB vector store.

Usage (from project root):
    python scripts/chroma_metrics.py
    python scripts/chroma_metrics.py --chroma-dir chroma_db --top 10
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="ChromaDB ingestion metrics")
    parser.add_argument("--chroma-dir", default="chroma_db", help="Path to the ChromaDB persist directory")
    parser.add_argument("--top", type=int, default=5, help="Number of top sources to display")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    try:
        import chromadb
    except ImportError:
        print("chromadb is not installed. Run: pip install chromadb", file=sys.stderr)
        sys.exit(1)

    chroma_path = Path(args.chroma_dir)
    if not chroma_path.exists():
        print(f"Directory not found: {chroma_path}", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(chroma_path))
    collections = client.list_collections()

    if not collections:
        print("No collections found in the database.", file=sys.stderr)
        sys.exit(1)

    report = {}

    for col in collections:
        col_obj = client.get_collection(col.name)
        total_chunks = col_obj.count()

        all_data = col_obj.get(include=["metadatas", "documents"])
        metadatas = all_data.get("metadatas") or []
        documents = all_data.get("documents") or []

        # --- Source / file breakdown ---
        sources = [
            m.get("source") or m.get("file_path") or m.get("filename") or "unknown"
            for m in metadatas
        ]
        source_counts = Counter(sources)

        # --- Chunk size stats ---
        chunk_lengths = [len(d) for d in documents if d]
        avg_chunk = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        min_chunk = min(chunk_lengths) if chunk_lengths else 0
        max_chunk = max(chunk_lengths) if chunk_lengths else 0

        # --- Metadata key coverage ---
        meta_keys: Counter = Counter()
        for m in metadatas:
            meta_keys.update(m.keys())

        # --- Optional enrichment fields ---
        with_summary = sum(1 for m in metadatas if m.get("summary"))
        with_subject = sum(1 for m in metadatas if m.get("subject"))

        report[col.name] = {
            "total_chunks": total_chunks,
            "unique_sources": len(source_counts),
            "top_sources": source_counts.most_common(args.top),
            "chunk_size": {
                "avg_chars": round(avg_chunk, 1),
                "min_chars": min_chunk,
                "max_chars": max_chunk,
            },
            "metadata_key_coverage": dict(meta_keys.most_common()),
            "enrichment": {
                "chunks_with_summary": with_summary,
                "chunks_with_subject": with_subject,
            },
        }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    for col_name, m in report.items():
        print(f"\n{'='*60}")
        print(f"  Collection : {col_name}")
        print(f"{'='*60}")
        print(f"  Total chunks     : {m['total_chunks']}")
        print(f"  Unique sources   : {m['unique_sources']}")

        print(f"\n  Top {args.top} sources by chunk count:")
        for src, count in m["top_sources"]:
            print(f"    {count:>5}  {src}")

        cs = m["chunk_size"]
        print(f"\n  Chunk size (chars):")
        print(f"    avg : {cs['avg_chars']}")
        print(f"    min : {cs['min_chars']}")
        print(f"    max : {cs['max_chars']}")

        print(f"\n  Metadata key coverage (out of {m['total_chunks']} chunks):")
        for key, count in m["metadata_key_coverage"].items():
            pct = count / m["total_chunks"] * 100 if m["total_chunks"] else 0
            print(f"    {key:<20} {count:>5}  ({pct:.0f}%)")

        enr = m["enrichment"]
        print(f"\n  Enrichment fields:")
        print(f"    summary  : {enr['chunks_with_summary']}")
        print(f"    subject  : {enr['chunks_with_subject']}")

    print()


if __name__ == "__main__":
    main()
