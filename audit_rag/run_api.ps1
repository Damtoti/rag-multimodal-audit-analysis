# Script de démarrage de l'API Audit RAG
$ErrorActionPreference = "Stop"

# Chemin du script pour un démarrage fiable depuis n'importe quel répertoire
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$VENV_PATH = Join-Path $SCRIPT_DIR "..\.venv\Scripts\python.exe"
$PYTHONPATH = Join-Path $SCRIPT_DIR "src"

# Définir les variables d'environnement
$env:PYTHONPATH = $PYTHONPATH
$env:PYTHONUNBUFFERED = "1"

Set-Location $SCRIPT_DIR

Write-Host "🚀 Audit RAG API - Démarrage" -ForegroundColor Green
Write-Host "📁 Répertoire: $((Get-Location).Path)" -ForegroundColor Cyan
Write-Host "🐍 Python: $VENV_PATH" -ForegroundColor Cyan
Write-Host "📦 PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Cyan
Write-Host ""

# Lancer uvicorn
Write-Host "⏳ Démarrage d'uvicorn..." -ForegroundColor Yellow
& $VENV_PATH -m uvicorn audit_rag.api:app `
    --host 127.0.0.1 `
    --port 8000 `
    --reload `
    --log-level info
