"""Tests unitaires pour le système RAG d\'audit."""
from unittest.mock import MagicMock, patch
 
import pytest
from langchain.schema import Document
 
from audit_rag.generator import AuditRAGGenerator
from audit_rag.retriever import AuditRetriever
 
 
@pytest.fixture
def mock_docs() -> list[Document]:
    return [
        Document(
            page_content="Le chiffre d\'affaires a augmenté de 12% en 2023.",
            metadata={"type": "text", "source": "rapport.pdf", "page": 5},
        ),
        Document(
            page_content="Ce tableau présente les résultats nets 2021-2023.",
            metadata={"type": "table", "source": "rapport.pdf", "page": 8},
        ),
    ]
 
 
@pytest.fixture
def mock_retriever(mock_docs: list[Document]) -> AuditRetriever:
    retriever = MagicMock(spec=AuditRetriever)
    retriever.retrieve.return_value = mock_docs
    return retriever
 
 
@pytest.fixture
def mock_generator(mock_retriever: AuditRetriever) -> AuditRAGGenerator:
    gen = AuditRAGGenerator(mock_retriever)
    with patch.object(gen.llm, "invoke") as mock_llm:
        mock_llm.return_value = MagicMock(
            content="Le chiffre d\'affaires a progressé de 12% (p.5)."
        )
        yield gen
 
 
def test_answer_returns_structure(mock_generator: AuditRAGGenerator) -> None:
    result = mock_generator.answer("Quelle est la croissance du CA ?")
    assert "question" in result
    assert "answer"   in result
    assert "source_docs" in result
    assert "metadata" in result
 
 
def test_answer_metadata_has_types(mock_generator: AuditRAGGenerator) -> None:
    result = mock_generator.answer("Y a-t-il des tableaux financiers ?")
    types = result["metadata"]["types"]
    assert "text"  in types
    assert "table" in types
 
 
def test_format_context_includes_source(mock_docs: list[Document]) -> None:
    ctx = AuditRAGGenerator._format_context(mock_docs)
    assert "rapport.pdf" in ctx
    assert "p.5" in ctx or "5" in ctx
