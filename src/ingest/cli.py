import argparse
import logging
import sys

from ingest.config import get_settings
from ingest.indexer import build_index
from ingest.loaders.factory import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument("--path", required=True, help="Path to a file or directory")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    settings = get_settings()

    try:
        documents = load_documents(args.path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not documents:
        print("No supported documents found.", file=sys.stderr)
        sys.exit(1)

    build_index(documents, settings)
    print(f"Done. Indexed {len(documents)} document(s) into {settings.chroma_persist_dir}/")


if __name__ == "__main__":
    main()
