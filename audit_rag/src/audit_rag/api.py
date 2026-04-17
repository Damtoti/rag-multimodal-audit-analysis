"""API FastAPI REST pour le système RAG d\'audit."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
 
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from audit_rag.config import get_settings
from audit_rag.extractor import PDFExtractor
from audit_rag.generator import AuditRAGGenerator
from audit_rag.retriever import AuditRetriever
from audit_rag.vectorstore import AuditVectorStore
 
logger = logging.getLogger(__name__)
cfg    = get_settings()
 
# ── État global de l\'application ─────────────────────────
_store:     Optional[AuditVectorStore] = None
_retriever: Optional[AuditRetriever]   = None
_generator: Optional[AuditRAGGenerator] = None
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore
    global _store, _retriever, _generator
    logger.info("Démarrage — chargement du vector store...")
    _store = AuditVectorStore()
    try:
        _store.load()
    except Exception:
        logger.warning("Aucun index trouvé — à créer via POST /ingest")
    _retriever = AuditRetriever(_store)
    _generator = AuditRAGGenerator(_retriever)
    yield
    logger.info("Arrêt de l\'application")
 
 
app = FastAPI(
    title="Audit RAG API",
    description="Analyse multi-modale de rapports d\'audit financier",
    version="0.1.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# ── Schémas Pydantic ─────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    k: int = 6
    use_mmr: bool = True
 
 
class QueryResponse(BaseModel):
    question: str
    answer: str
    source_docs: list
    metadata: dict
 
 
class IngestResponse(BaseModel):
    filename: str
    elements_extracted: int
    status: str
 
 
class HealthResponse(BaseModel):
    status: str
    index_size: int
 
 
# ── Endpoints ────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    count = 0
    if _store and _store._store:
        count = _store._store._collection.count()
    return HealthResponse(status="ok", index_size=count)
 
 
@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """Ingère un rapport PDF et l\'ajoute à l\'index."""
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")
 
    dest = cfg.data_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)
 
    extractor = PDFExtractor()
    elements  = extractor.process(dest)
 
    if _store is None:
        raise HTTPException(status_code=503, detail="Vector store non disponible")
 
    _store.build(elements)
    return IngestResponse(
        filename=file.filename,
        elements_extracted=len(elements),
        status="indexed",
    )
 
 
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Interroge le système RAG sur les rapports indexés."""
    if _generator is None or _store is None or _store._store is None:
        raise HTTPException(status_code=503, detail="Aucun document indexé")
 
    result = _generator.answer(req.question, k=req.k, use_mmr=req.use_mmr)
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        metadata=result["metadata"],
    )