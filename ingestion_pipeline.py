"""Load, chunk, and store local text documents for the RAG ingestion pipeline."""

from pathlib import Path
import shutil

from dotenv import load_dotenv

# Chroma stores document chunks and their embedding vectors on disk.
from langchain_chroma import Chroma
# DirectoryLoader finds files; TextLoader reads each file as plain text.
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# A Document stores text in page_content and details such as its source in metadata.
from langchain_core.documents import Document
# OpenAI embeddings convert text into vectors; reads OPENAI_API_KEY from the environment.
from langchain_openai import OpenAIEmbeddings
# This splitter tries paragraph and line boundaries before splitting finer text.
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Resolve paths from this file so the script works from any terminal directory.
MAIN_FOLDER = Path(__file__).resolve().parent
DOCS_FOLDER = MAIN_FOLDER / "docs"

# Load local API keys without replacing variables already set in the environment.
load_dotenv(MAIN_FOLDER / ".env")


def load_documents(docs_folder: Path = DOCS_FOLDER) -> list[Document]:
    """Load UTF-8 .txt files from the docs folder and its subfolders."""
    # Fail early with a clear message if the input directory does not exist.
    if not docs_folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {docs_folder}")

    loader = DirectoryLoader(
        path=str(docs_folder),
        # ** includes nested folders; *.txt limits loading to text files.
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    # This reads local files only; embedding requests happen later.
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).", flush=True)
    for document in documents:
        print(
            f"  - {document.metadata['source']}: "
            f"{len(document.page_content):,} characters",
            flush=True,
        )
    return documents


def splittingchunks(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 0
) -> list[Document]:
    """Split documents into overlapping text chunks, preserving metadata.

    chunk_size is the maximum number of characters per chunk (default: 1000).
    chunk_overlap is the target number of shared characters between adjacent
    chunks to help retain context (default: 0, so chunks do not overlap).
    """
    print(
        f"Chunk settings: size={chunk_size} characters, "
        f"overlap={chunk_overlap} characters.",
        flush=True,
    )
    # Prefer natural text boundaries while keeping chunks within the size limit.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    # Return new Documents for the chunks, carrying over each source's metadata.
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunk(s) for embedding.", flush=True)
    return chunks


def createvectorstore(
    chunks: list[Document], persistdirectory: str = "db/chroma_db"
) -> Chroma:
    """Embed chunks and save them in a Chroma store using cosine distance.

    Requires OPENAI_API_KEY. Relative storage paths are resolved from this file.
    Calling this again adds chunks to the existing store.
    """
    if not chunks:
        raise ValueError("Cannot create a vector store without chunks.")

    # Keep the database location consistent when running from another directory.
    persist_path = Path(persistdirectory)
    if not persist_path.is_absolute():
        persist_path = MAIN_FOLDER / persist_path

    print(f"Vector store directory: {persist_path}", flush=True)
    print("Similarity metric: cosine.", flush=True)
    # Constructing this object prepares the client; it does not embed text yet.
    print("Preparing OpenAI embedding model: text-embedding-3-small...", flush=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # from_documents sends the chunk text to OpenAI, then stores the returned
    # vectors with the original text and metadata in the local Chroma database.
    print(
        f"Embedding {len(chunks)} chunk(s) through OpenAI and saving to Chroma...",
        flush=True,
    )
    StoredVector = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        # Rank vectors by their direction: cosine distance = 1 - cosine similarity.
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    # Print success only after embedding and persistence both finish.
    print(f"Vector store saved successfully to {persist_path}.", flush=True)
    return StoredVector


def delete_database() -> None:
    """Delete the default Chroma database so ingestion starts with a fresh store."""
    db_directory = MAIN_FOLDER / "db" / "chroma_db"
    # Verify the deletion target; refuse paths redirected by a symlink or junction.
    if db_directory.resolve() != MAIN_FOLDER / "db" / "chroma_db":
        raise ValueError(f"Database deletion path is redirected: {db_directory}")
    if db_directory.exists():
        print(f"Deleting existing database: {db_directory}...", flush=True)
        shutil.rmtree(db_directory)
        print("Database deleted. This run will create a fresh store.", flush=True)
    else:
        print("No existing database found. This run will create a fresh store.", flush=True)


def main() -> None:
    """Run the ingestion stages in order: load, split, then embed and store."""
    # Flush progress messages immediately so they appear before slow API calls
    # and any error traceback, including when terminal output is redirected.
    print("Starting document ingestion pipeline.", flush=True)

    # Comment out this call to keep the existing database between runs.
    delete_database()

    # Step 1: Load the original documents from the local docs directory.
    print(f"\n[1/3] Loading text documents from {DOCS_FOLDER}...", flush=True)
    documents = load_documents()
    # Step 2: Split them using the defaults: 1000 characters and zero overlap.
    print("\n[2/3] Splitting documents into chunks...", flush=True)
    chunks = splittingchunks(documents)
    # Step 3: Embed the chunks and persist them in the default Chroma directory.
    print("\n[3/3] Creating the persistent vector store...", flush=True)
    StoredVector = createvectorstore(chunks)
    # StoredVector is the returned Chroma object, ready for similarity searches.
    print(
        f"\nIngestion complete: {len(documents)} document(s), "
        f"{len(chunks)} chunk(s) processed and stored.",
        flush=True,
    )


# Run the pipeline when executed directly; importing this module does not run it.
if __name__ == "__main__":
    main()
