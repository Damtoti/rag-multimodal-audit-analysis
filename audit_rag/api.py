"""Compatibility ASGI entrypoint for deployments importing audit_rag.api from repo root."""

from .src.audit_rag.api import app  # type: ignore F401
