"""Check project paths and ingestion without API calls or the real database."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from langchain_core.documents import Document

from rag_sample import ingestion_pipeline as pipeline


class IngestionPipelineTests(TestCase):
    def test_project_paths(self):
        project_root = Path(__file__).resolve().parent.parent
        self.assertEqual(pipeline.MAIN_FOLDER, project_root)
        self.assertEqual(pipeline.DOCS_FOLDER, project_root / "docs")

    def test_load_and_split_nested_text_files_preserves_source(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            source = nested / "sample.txt"
            source.write_text("a" * 2500, encoding="utf-8")
            (root / "ignored.md").write_text("not an input", encoding="utf-8")

            documents = pipeline.load_documents(root)
            chunks = pipeline.splittingchunks(documents)

            self.assertEqual(len(documents), 1)
            self.assertEqual([len(chunk.page_content) for chunk in chunks], [1000, 1000, 500])
            self.assertTrue(all(chunk.metadata["source"] == str(source) for chunk in chunks))

    def test_database_reset_only_deletes_the_database(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = root / "db" / "chroma_db"
            database.mkdir(parents=True)
            (database / "sample.sqlite3").write_text("sample", encoding="utf-8")
            sibling = root / "db" / "keep.txt"
            sibling.write_text("keep", encoding="utf-8")
            with patch.object(pipeline, "MAIN_FOLDER", root), redirect_stdout(StringIO()):
                pipeline.delete_database()
                pipeline.delete_database()  # A missing database is also supported.
            self.assertFalse(database.exists())
            self.assertEqual(sibling.read_text(encoding="utf-8"), "keep")

    def test_database_reset_rejects_redirected_paths(self):
        with patch.object(Path, "resolve", return_value=pipeline.MAIN_FOLDER.parent / "outside"):
            with patch.object(pipeline.shutil, "rmtree") as delete:
                with self.assertRaises(ValueError):
                    pipeline.delete_database()
                delete.assert_not_called()

    def test_main_resets_before_loading_and_passes_chunks_to_storage(self):
        documents = [Document(page_content="Sample text", metadata={"source": "sample.txt"})]
        stages = []

        def load():
            stages.append("load")
            return documents

        def store(chunks):
            stages.append("store")
            self.assertEqual(chunks[0].page_content, "Sample text")
            return object()

        with (
            patch.object(pipeline, "delete_database", side_effect=lambda: stages.append("reset")),
            patch.object(pipeline, "load_documents", side_effect=load),
            patch.object(pipeline, "createvectorstore", side_effect=store),
            redirect_stdout(StringIO()) as output,
        ):
            pipeline.main()
        self.assertEqual(stages, ["reset", "load", "store"])
        self.assertIn("Ingestion complete", output.getvalue())
