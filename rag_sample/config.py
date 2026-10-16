"""Project paths and local environment configuration."""

from pathlib import Path

from dotenv import load_dotenv

# Keep data and secrets at the project root, outside the Python package.
MAIN_FOLDER = Path(__file__).resolve().parent.parent
DOCS_FOLDER = MAIN_FOLDER / "docs"

# Existing environment variables take precedence over the project's .env file.
load_dotenv(MAIN_FOLDER / ".env")
