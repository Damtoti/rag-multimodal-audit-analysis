"""Gestion du vector store ChromaDB pour les rapports d'audit."""
import logging
from typing import Optional
 
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
 
from audit_rag.config import Settings, get_settings
from audit_rag.extractor import ExtractedElement
 
logger = logging.getLogger(__name__)
 
COLLECTION_NAME = "audit_reports"
 
 
class AuditVectorStore:
    """Index vectoriel ChromaDB pour documents d'audit multi-modaux."""
 
    def __init__(self, settings: Optional[Settings] = None):
        self.cfg = settings or get_settings()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.splitter   = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            separators=["\\n\\n", "\\n", ".", "!", "?", ";", " ", ""],
        )
        self._store: Optional[Chroma] = None
 
    # ── Build / Load ─────────────────────────────────────────
    def build(self, elements: list[ExtractedElement]) -> None:
        docs = self._prepare_documents(elements)
        logger.info("Indexation de %d documents...", len(docs))
        self._store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=str(self.cfg.persist_dir),
            collection_name=COLLECTION_NAME,
        )
        self._store.persist()
        logger.info("Index persisté dans %s", self.cfg.persist_dir)
 
    def load(self) -> None:
        self._store = Chroma(
            persist_directory=str(self.cfg.persist_dir),
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            collection_name=COLLECTION_NAME,
        )
        count = self._store._collection.count()
        logger.info("Index chargé : %d documents", count)
 
    # ── Recherche ────────────────────────────────────────────
    def similarity_search(
        self,
        query: str,
        k: int = 6,
        filter_type: Optional[str] = None,
    ) -> list[Document]:
        self._assert_ready()
        flt = {"type": filter_type} if filter_type else None
        return self._store.similarity_search(query, k=k, filter=flt)  # type: ignore
 
    def mmr_search(self, query: str, k: int = 6, fetch_k: int = 20) -> list[Document]:
        self._assert_ready()
        return self._store.max_marginal_relevance_search(  # type: ignore
            query, k=k, fetch_k=fetch_k
        )
 
    # ── Helpers ─────────────────────────────────────────────
    def _prepare_documents(self, elements: list[ExtractedElement]) -> list[Document]:
        docs: list[Document] = []
        for elem in elements:
            base_meta = {
                "type":   elem.element_type,
                "source": elem.source_file,
                "page":   elem.page_number,
                **{k: v for k, v in elem.metadata.items()
                   if k != "clip_embedding"},
            }
            if elem.element_type == "text":
                for i, chunk in enumerate(self.splitter.split_text(elem.content)):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={**base_meta, "chunk_index": i},
                    ))
            else:
                docs.append(Document(page_content=elem.content, metadata=base_meta))
        return docs
 
    def _assert_ready(self) -> None:
        if self._store is None:
            raise RuntimeError("Vector store non initialisé. Appeler build() ou load().")
