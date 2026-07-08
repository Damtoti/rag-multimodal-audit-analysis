"""API FastAPI REST pour le système RAG d\'audit."""
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
 
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile, status
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

# ── Job tracking ─────────────────────────────────────────
_jobs: dict[str, dict[str, Any]] = {}

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


class IngestJobResponse(BaseModel):
    job_id: str
    filename: str
    status: str


class IngestJobStatus(BaseModel):
    job_id: str
    filename: str
    status: str          # "pending" | "done" | "error"
    elements_extracted: int
    detail: str


class HealthResponse(BaseModel):
    status: str
    index_size: int


# ── Ingest background worker ─────────────────────────────
def _run_ingest_sync(job_id: str, dest: Path, filename: str) -> None:
    """CPU-bound extraction + indexing executed in thread pool."""
    from audit_rag.extractor import PDFExtractor

    _jobs[job_id] = {"filename": filename, "status": "pending", "elements_extracted": 0, "detail": ""}
    try:
        extractor = PDFExtractor()
        elements  = extractor.process(dest)

        if not elements:
            _jobs[job_id].update(
                status="error",
                detail=(
                    "Aucun contenu extractible dans ce PDF. "
                    "Fichier scanné/image ou vide. "
                    "Activez ENABLE_IMAGE_EXTRACTION=true pour les PDF scannés."
                ),
            )
            return

        if _store is None:
            _jobs[job_id].update(status="error", detail="Vector store non disponible")
            return

        _store.build(elements)
        _jobs[job_id].update(status="done", elements_extracted=len(elements), detail="")
        logger.info("Ingest job %s done: %d elements", job_id, len(elements))

    except Exception as exc:
        logger.exception("Ingest job %s failed", job_id)
        _jobs[job_id].update(status="error", detail=f"{type(exc).__name__}: {exc}")


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


@app.post("/ingest", response_model=IngestJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> IngestJobResponse:
    """Lance l\'ingestion d\'un PDF en tâche de fond — répond immédiatement."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")

    filename = Path(file.filename).name
    dest = cfg.data_dir / filename
    max_bytes = cfg.max_upload_mb * 1024 * 1024
    total_written = 0

    # Stream to disk to avoid loading large PDFs into memory.
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > max_bytes:
                    out.close()
                    try:
                        dest.unlink(missing_ok=True)
                    except Exception:
                        logger.warning("Impossible de supprimer le fichier trop volumineux: %s", dest)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Fichier trop volumineux (> {cfg.max_upload_mb} MB)",
                    )
                out.write(chunk)
    finally:
        await file.close()

    if total_written == 0:
        raise HTTPException(status_code=400, detail="Le fichier uploadé est vide")

    job_id = str(uuid.uuid4())[:8]
    # Run after returning HTTP 202 to avoid gateway timeout/502 on long ingest startup.
    background_tasks.add_task(_run_ingest_sync, job_id, dest, filename)

    return IngestJobResponse(job_id=job_id, filename=filename, status="pending")


@app.get("/ingest/{job_id}", response_model=IngestJobStatus)
async def ingest_status(job_id: str) -> IngestJobStatus:
    """Retourne l\'état d\'un job d\'ingestion."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} inconnu")
    return IngestJobStatus(job_id=job_id, **job)
 
 
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
            detail=(
                "Erreur OpenAI: "
                f"{type(exc).__name__}: {str(exc)[:260]}"
            ),
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