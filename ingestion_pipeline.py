"""Load local text documents for the RAG ingestion pipeline."""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


# Resolve paths from this file so the script works from any terminal directory.
MAIN_FOLDER = Path(__file__).resolve().parent
DOCS_FOLDER = MAIN_FOLDER / "docs"


def load_documents(docs_folder: Path = DOCS_FOLDER) -> list[Document]:
    """Load UTF-8 .txt files from the docs folder and its subfolders."""
    if not docs_folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {docs_folder}")

    loader = DirectoryLoader(
        path=str(docs_folder),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def main() -> None:
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s) from {DOCS_FOLDER}")
    for document in documents:
        print(
            f"- {document.metadata['source']}: "
            f"{len(document.page_content):,} characters"
        )


if __name__ == "__main__":
    main()
