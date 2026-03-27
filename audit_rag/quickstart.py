# ════════════════════════════════════════════════════════════
# QUICK START (imprimé si exécuté directement)
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║   RAG MULTI-MODAL — AUDIT FINANCIER — QUICK START VSCODE    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Créer le projet Poetry :                                 ║
║     poetry new audit-rag                                     ║
║     cd audit-rag                                             ║
║     # Copier les fichiers src/, tests/, scripts/             ║
║     # Copier pyproject.toml                                  ║
║                                                              ║
║  2. Installer les dépendances :                              ║
║     poetry install                                           ║
║                                                              ║
║  3. Configurer les clés API :                                ║
║     cp .env.example .env                                     ║
║     # Remplir OPENAI_API_KEY dans .env                       ║
║                                                              ║
║  4. Ingérer les rapports PDF :                               ║
║     mkdir -p data/reports                                    ║
║     cp vos_rapports.pdf data/reports/                        ║
║     poetry run python scripts/ingest.py                      ║
║                                                              ║
║  5. Lancer l'API REST :                                      ║
║     poetry run uvicorn audit_rag.api:app --reload            ║
║     # → http://localhost:8000/docs                           ║
║                                                              ║
║  6. Interroger via curl :                                    ║
║     curl -X POST http://localhost:8000/query \\              ║
║       -H "Content-Type: application/json" \\                 ║
║       -d '{"question":"Risques financiers du rapport?"}'     ║
║                                                              ║
║  7. Évaluation RAGAS :                                       ║
║     poetry run python scripts/evaluate.py                    ║
║                                                              ║
║  8. Tests :                                                  ║
║     poetry run pytest tests/ -v --cov=src                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")