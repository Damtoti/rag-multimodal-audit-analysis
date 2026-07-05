"""API FastAPI REST pour le système RAG d\'audit."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
 
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAIError
 
from audit_rag.config import get_settings
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
    logger.info("Démarrage — initialisation des composants...")
    _store = AuditVectorStore()
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

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Servir la page d'accueil."""
    try:
        html_path = Path(__file__).parent / "static" / "index.html"
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error("Erreur serveur racine: %s", e)
        return "<h1>Erreur : impossible de charger la page</h1>"


# ── Schémas Pydantic ─────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    k: int = 6
    use_mmr: bool = True
 
 
class QueryResponse(BaseModel):
    question: str
    answer: str
    source_docs: list[dict[str, Any]]
    metadata: dict[str, Any]
 
 
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
    if _store:
        if _store._store is None:
            try:
                _store.load()
            except Exception:
                logger.info("Aucun index chargé pour /health")
        if _store._store:
            count = _store._store._collection.count()
    return HealthResponse(status="ok", index_size=count)
 
 
@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """Ingère un rapport PDF et l\'ajoute à l\'index."""
    from audit_rag.extractor import PDFExtractor

    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")
 
    dest = cfg.data_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)
 
    try:
        extractor = PDFExtractor()
        elements  = extractor.process(dest)
    except Exception as exc:
        logger.exception("Erreur ingestion/extraction pour %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Echec extraction PDF: {type(exc).__name__}: {exc}",
        )
 
    if _store is None:
        raise HTTPException(status_code=503, detail="Vector store non disponible")
 
    try:
        _store.build(elements)
    except Exception as exc:
        logger.exception("Erreur indexation vector store pour %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail=f"Echec indexation vectorielle: {type(exc).__name__}: {exc}",
        )
    return IngestResponse(
        filename=file.filename,
        elements_extracted=len(elements),
        status="indexed",
    )
 
 
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Interroge le système RAG sur les rapports indexés."""
    if _store is not None and _store._store is None:
        try:
            _store.load()
        except Exception:
            logger.info("Aucun index disponible pour /query")

    if _generator is None or _store is None or _store._store is None:
        raise HTTPException(status_code=503, detail="Aucun document indexé")
 
    try:
        result = _generator.answer(req.question, k=req.k, use_mmr=req.use_mmr)
    except OpenAIError as exc:
        logger.error("OpenAI error during query: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Erreur de requête OpenAI : vérifiez votre clé API et votre configuration.",
        )
    except Exception:
        logger.exception("Erreur interne lors du traitement de la requête")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")
 
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        source_docs=result["source_docs"],
        metadata=result["metadata"],
    )