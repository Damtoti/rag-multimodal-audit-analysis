#!/usr/bin/env python
"""Script de démarrage de l'API Audit RAG avec configuration correcte."""
import os
import sys
from pathlib import Path

# Ajouter le répertoire src au chemin Python
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Définir les variables d'environnement
os.environ["PYTHONPATH"] = str(src_path)

# Importer et lancer uvicorn
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "audit_rag.api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", os.getenv("API_PORT", "8000"))),
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
        log_level="info",
    )
