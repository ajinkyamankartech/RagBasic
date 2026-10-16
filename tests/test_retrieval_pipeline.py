"""Test retrieval without paid API calls or touching the real vector database."""

import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest import TestCase
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from rag_sample import retrieval_pipeline as pipeline


class RetrievalPipelineTests(TestCase):
    def test_missing_database_is_not_created(self):
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "Run ingestion first"):
                pipeline.load_vectorstore(str(missing))
            self.assertFalse(missing.exists())

    def test_invalid_search_does_not_call_embedding_api(self):
        vectorstore = Mock()
        for query, k in [("   ", 3), ("whale", 0)]:
            with self.subTest(query=query, k=k), self.assertRaises(ValueError):
                pipeline.retrieve_chunks(query, vectorstore, k)
        vectorstore.similarity_search.assert_not_called()

    def test_non_cosine_collection_is_rejected_before_search(self):
        vectorstore = Mock()
        vectorstore._collection.configuration = {"hnsw": {"space": "l2"}}
        with self.assertRaisesRegex(ValueError, "cosine is required"):
            pipeline.retrieve_chunks("Who is Ahab?", vectorstore)
        vectorstore.similarity_search.assert_not_called()

    def test_cli_prints_matching_text_and_source(self):
        chunks = [Document(page_content="Captain Ahab commands the Pequod.", metadata={"source": "sample.txt"})]
        with (
            patch.object(sys, "argv", ["retrieval", "Who is Ahab?", "--k", "1"]),
            patch.object(pipeline, "load_vectorstore") as load,
            patch.object(pipeline, "retrieve_chunks", return_value=chunks) as retrieve,
            redirect_stdout(StringIO()) as output,
        ):
            pipeline.main()
        retrieve.assert_called_once_with("Who is Ahab?", load.return_value, k=1)
        self.assertIn("sample.txt", output.getvalue())
        self.assertIn("Captain Ahab commands the Pequod.", output.getvalue())

    def test_cli_prompts_when_query_is_omitted(self):
        with (
            patch.object(sys, "argv", ["retrieval"]),
            patch("builtins.input", return_value="Who is Ahab?") as prompt,
            patch.object(pipeline, "load_vectorstore") as load,
            patch.object(pipeline, "retrieve_chunks", return_value=[]) as retrieve,
            redirect_stdout(StringIO()),
        ):
            pipeline.main()
        prompt.assert_called_once_with("Enter your question: ")
        retrieve.assert_called_once_with("Who is Ahab?", load.return_value, k=5)

    def test_empty_prompt_is_rejected_before_loading_database(self):
        with (
            patch.object(sys, "argv", ["retrieval"]),
            patch("builtins.input", return_value="   "),
            patch.object(pipeline, "load_vectorstore") as load,
            redirect_stderr(StringIO()) as errors,
            self.assertRaises(SystemExit) as error,
        ):
            pipeline.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Provide a nonempty question", errors.getvalue())
        load.assert_not_called()

    def test_missing_terminal_input_has_clear_error(self):
        with (
            patch.object(sys, "argv", ["retrieval"]),
            patch("builtins.input", side_effect=EOFError),
            patch.object(pipeline, "load_vectorstore") as load,
            redirect_stderr(StringIO()) as errors,
            self.assertRaises(SystemExit) as error,
        ):
            pipeline.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Pass a question as a command-line argument", errors.getvalue())
        load.assert_not_called()

    def test_persisted_cosine_search_reuses_document_embeddings(self):
        # Run Chroma in a child process so Windows releases SQLite/index handles
        # before the temporary directory is cleaned up by the parent process.
        code = dedent('''
            import sys
            from pathlib import Path
            from unittest.mock import patch
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            from langchain_core.embeddings import Embeddings
            from rag_sample import retrieval_pipeline as pipeline

            class TestEmbeddings(Embeddings):
                def embed_documents(self, texts):
                    # Cosine prefers whale's direction; L2 would prefer mountain.
                    return [[10., 0.] if text == "whale" else [1., 1.] for text in texts]
                def embed_query(self, text):
                    return [2., 0.]

            embedding = TestEmbeddings()
            path = Path(sys.argv[1]) / "populated"
            store = Chroma.from_documents(
                [Document(page_content="whale", metadata={"source": "sea.txt"}),
                 Document(page_content="mountain", metadata={"source": "land.txt"})],
                embedding=embedding,
                persist_directory=str(path),
                collection_configuration={"hnsw": {"space": "cosine"}},
            )
            original_ids = set(store.get()["ids"])
            with patch.object(pipeline, "OpenAIEmbeddings", return_value=embedding):
                with patch.object(embedding, "embed_documents", side_effect=AssertionError("Documents must not be re-embedded")):
                    loaded = pipeline.load_vectorstore(str(path))
                    results = pipeline.retrieve_chunks("sea creature", loaded, k=1)
                assert results[0].page_content == "whale"
                assert results[0].metadata["source"] == "sea.txt"
                assert set(loaded.get()["ids"]) == original_ids
                assert loaded._collection.configuration["hnsw"]["space"] == "cosine"

                empty_path = Path(sys.argv[1]) / "empty"
                Chroma(persist_directory=str(empty_path), embedding_function=embedding)
                try:
                    pipeline.load_vectorstore(str(empty_path))
                except ValueError as error:
                    assert "contains no chunks" in str(error)
                else:
                    raise AssertionError("An empty database must be rejected")
        ''')
        with TemporaryDirectory(prefix="rag-retrieval-test-") as directory:
            result = subprocess.run(
                [sys.executable, "-c", code, directory],
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "ANONYMIZED_TELEMETRY": "False"},
            )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
