"""Path and environment configuration for the RAG project."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# app/ -> Advanced_RAG_Walmart_project/ -> repo root
PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent

DATA_DIR = REPO_ROOT / "data"
PDF_PATH = DATA_DIR / "Walmart Annual Report 2025.pdf"
TEST_QUESTIONS = PACKAGE_DIR / "tests" / "test_questions.json"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "walmart_annual_report_2025")

load_dotenv(REPO_ROOT / ".env")

MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))
