"""Retrieval avec MMR et reranking optionnel (Cohere)."""
import logging
from typing import Optional
 
from langchain.schema import Document
 
from audit_rag.config import Settings, get_settings
from audit_rag.vectorstore import AuditVectorStore
 
logger = logging.getLogger(__name__)
 
 
class AuditRetriever:
    """Retrieval multi-modal avec MMR + reranking."""
 
    def __init__(
        self,
        store: AuditVectorStore,
        settings: Optional[Settings] = None,
    ):
        self.store = store
        self.cfg   = settings or get_settings()
        self._cohere = None
        if self.cfg.cohere_api_key:
            try:
                import cohere
                self._cohere = cohere.Client(self.cfg.cohere_api_key)
                logger.info("Cohere reranking activé")
            except ImportError:
                logger.warning("cohere non installé — pip install cohere")
 
    def retrieve(
        self,
        query: str,
        k: int = 6,
        use_mmr: bool = True,
        rerank: bool = True,
    ) -> list[Document]:
        """Récupère les documents pertinents avec MMR et reranking optionnel."""
        fetch_k = self.cfg.fetch_k
 
        if use_mmr:
            docs = self.store.mmr_search(query, k=fetch_k, fetch_k=fetch_k * 3)
        else:
            docs = self.store.similarity_search(query, k=fetch_k)
 
        if rerank and self._cohere and len(docs) > k:
            docs = self._cohere_rerank(query, docs, top_n=k)
        else:
            docs = docs[:k]
 
        logger.debug("Retrieval '%s' → %d docs (types: %s)",
                     query[:50],
                     len(docs),
                     [d.metadata.get("type") for d in docs])
        return docs
 
    def _cohere_rerank(
        self,
        query: str,
        docs: list[Document],
        top_n: int,
    ) -> list[Document]:
        results = self._cohere.rerank(
            query=query,
            documents=[d.page_content for d in docs],
            top_n=top_n,
            model="rerank-multilingual-v3.0",
        )
        return [docs[r.index] for r in results.results]