# RAGSample

A Python sample that loads UTF-8 text documents, splits them into chunks, and
stores OpenAI embeddings in a local Chroma database using cosine distance.

## Project layout

```text
RAGSample/
|-- rag_sample/                 # Application package
|   |-- __init__.py
|   |-- __main__.py             # python -m rag_sample
|   |-- config.py               # Project paths and .env loading
|   |-- ingestion_pipeline.py   # Loading, chunking, storage, and database reset
|   `-- retrieval_pipeline.py   # Load stored vectors and search for a question
|-- tests/                      # Offline regression tests
|-- docs/                       # Input .txt documents (including subfolders)
|-- db/                         # Generated Chroma data; ignored by Git
|-- .env                        # Local API key; ignored by Git
|-- .env.example                # Safe configuration template
|-- .gitignore
|-- .gitattributes
|-- pyproject.toml              # Package metadata, dependencies, and CLI entry point
`-- README.md
```

This project uses a [flat Python package layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
Document inputs and generated
databases stay outside the application package. Project paths are anchored to
the checkout, so changing the working directory does not change the data paths.
This sample is intended to run from a source checkout or an editable install.

## Setup

Use Python 3.12 or newer. From the project root in PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -e .
```

For a new checkout, copy `.env.example` to `.env` and set `OPENAI_API_KEY` there.
If `.env` already exists, keep that file. You can also set the key in the process
environment; it takes precedence over `.env`.

## Run ingestion

```powershell
.\venv\Scripts\python.exe -m rag_sample
```

After installation, `rag-ingest` is also available in the virtual environment.

The pipeline:

1. Calls `delete_database()` to remove `db/chroma_db` if it exists.
2. Loads `.txt` files from `docs/` with their source metadata.
3. Splits them into chunks of up to 1,000 characters, with zero overlap.
4. Embeds the chunks with `text-embedding-3-small` and saves them in Chroma.

To preserve the existing database, comment out `delete_database()` in
`rag_sample/ingestion_pipeline.py` inside `main()`. Running ingestion with the
deletion call disabled still embeds and appends the supplied chunks; it does not
skip documents already stored. Each ingestion run makes billable OpenAI API calls.

Commit source files and configuration templates. Keep `.env`, `venv/`, and `db/`
local; `.gitignore` excludes them.

## Run retrieval

After ingestion has completed, search the existing database:

```powershell
.\venv\Scripts\python.exe -m rag_sample.retrieval_pipeline "Who is Captain Ahab?" --k 5
```

You can also run it without a question argument and enter your question when
the terminal displays `Enter your question:`:

```powershell
.\venv\Scripts\python.exe -m rag_sample.retrieval_pipeline
```

This returns up to five matching chunks by default and prints their source paths.
Retrieval uses the same
embedding model as ingestion and the saved collection's cosine distance metric.
The search checks that the saved collection uses cosine before embedding the
question; a collection with another metric must be rebuilt with cosine first.
It embeds only the question and reuses document vectors already stored in Chroma.
It does not delete the database or ingest the book again. An absent or empty
database raises an error asking you to run ingestion first.

You can select another existing database with `--persist-directory db/chroma_db`.
After installation, `rag-retrieve "Who is Captain Ahab?"` is also available in
the virtual environment. This stage returns matching passages; it does not
generate an answer with a language model.

## Offline checks

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary directories and mock embedding/storage calls. They do not
delete the real database or make OpenAI API calls.
