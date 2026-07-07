"""Compatibility ASGI entrypoint for deployments importing audit_rag.api from repo root."""

import sys


if sys.version_info >= (3, 14):
	raise RuntimeError(
		"Python 3.14+ is not supported by this project dependencies (LangChain/Pydantic v1 path). "
		"Use Python 3.10-3.13, recreate the virtual environment, and reinstall dependencies."
	)

from .src.audit_rag.api import app  # type: ignore F401
