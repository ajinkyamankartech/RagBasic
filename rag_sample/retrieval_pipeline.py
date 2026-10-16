"""Retrieve relevant document chunks from the existing Chroma database."""

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag_sample.config import MAIN_FOLDER


def load_vectorstore(persistdirectory: str = "db/chroma_db") -> Chroma:
    """Open the existing vector store without ingesting documents again."""
    # Use the same project-root path resolution as the ingestion pipeline.
    persist_path = Path(persistdirectory)
    if not persist_path.is_absolute():
        persist_path = MAIN_FOLDER / persist_path
    if not (persist_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"Chroma database not found at {persist_path}. Run ingestion first."
        )

    print(f"Loading vector store from {persist_path}...", flush=True)
    # Queries must use the same embedding model used to store the documents.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        # An incorrect path or missing collection must not create an empty store.
        create_collection_if_not_exists=False,
    )
    # A previous failed ingestion may have created a database without any chunks.
    if not vectorstore.get(limit=1)["ids"]:
        raise ValueError("The vector store contains no chunks. Run ingestion first.")
    print("Existing vector store loaded. Document embeddings will be reused.", flush=True)
    return vectorstore


def retrieve_chunks(query: str, vectorstore: Chroma, k: int = 5) -> list[Document]:
    """Return up to k stored chunks using Chroma's cosine similarity search."""
    if not query.strip():
        raise ValueError("The query must not be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    # Chroma's distance metric belongs to the saved collection, not each query.
    # Check the real index configuration before sending the question to OpenAI.
    configuration = vectorstore._collection.configuration
    distance_metric = (configuration.get("hnsw") or {}).get("space")
    if distance_metric != "cosine":
        raise ValueError(
            f"The Chroma collection uses {distance_metric!r}; cosine is required. "
            "Rebuild the collection with cosine configured in the ingestion pipeline."
        )

    print("Search metric: Chroma cosine distance (1 - cosine similarity).", flush=True)
    print(f"Embedding the question and searching for the top {k} chunk(s)...", flush=True)
    # Only the question is sent to OpenAI. Chroma compares its vector against
    # stored document vectors using the metric set during ingestion (cosine).
    chunks = vectorstore.similarity_search(query=query, k=k)
    print(f"Retrieved {len(chunks)} relevant chunk(s).", flush=True)
    return chunks


def main() -> None:
    """Load the database, search for a question, and print the matching text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query", nargs="?", help="Question to search for; prompts when omitted."
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Number of chunks to return (default: 5)."
    )
    parser.add_argument(
        "--persist-directory", default="db/chroma_db", help="Existing Chroma database path."
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1.")

    # Running the file with the editor's Run button supplies no query argument.
    # Ask for one in the terminal while keeping command-line queries supported.
    query = args.query
    if query is None:
        try:
            query = input("Enter your question: ")
        except EOFError:
            parser.error("No question provided. Pass a question as a command-line argument.")
    if not query.strip():
        parser.error("Provide a nonempty question.")

    # Step 1: Open the stored collection. Retrieval never calls delete_database().
    print("[1/2] Loading the existing database...", flush=True)
    vectorstore = load_vectorstore(args.persist_directory)
    # Step 2: Embed the question and find matching document chunks.
    print("\n[2/2] Retrieving relevant chunks...", flush=True)
    chunks = retrieve_chunks(query, vectorstore, k=args.k)
    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Result {index} | Source: {chunk.metadata.get('source', 'Unknown')} ---")
        print(chunk.page_content)


if __name__ == "__main__":
    main()
