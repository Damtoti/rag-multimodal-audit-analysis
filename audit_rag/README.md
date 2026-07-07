# Audit RAG - Analyse Multimodale de Rapports d'Audit

Un système RAG (Retrieval-Augmented Generation) avancé pour l'analyse intelligente et multimodale de rapports d'audit.

##  Objectifs

- Extraire et indexer automatiquement les données des rapports d'audit (PDF, images, tableaux)
- Récupérer les informations pertinentes selon les requêtes utilisateur
- Générer des analyses synthétiques et contextualisées
- Évaluer la qualité et la pertinence des réponses générées

##  Démarrage rapide

### Prérequis

- Python 3.10 a 3.13
- Docker et Docker Compose (optionnel)
- API Key OpenAI

### Installation

1. **Installer les dépendances**
```bash
poetry install
```

2. **Configurer les variables d'environnement**
```bash
cp .env.example .env
# Éditer .env avec vos configurations
```

3. **Démarrer les services**
```bash
docker-compose up -d
```

### Utilisation

**Ingestion de documents :**
```bash
poetry run python scripts/ingest.py --path documents/
```

**Exécuter le quickstart :**
```bash
poetry run python quickstart.py
```

**API RESTful :**
```bash
poetry run uvicorn src.audit_rag.api:app --reload
```

##  Structure du projet

```
audit_rag/
├── src/audit_rag/          # Code source principal
│   ├── api.py             # API FastAPI
│   ├── config.py          # Configuration
│   ├── extractor.py       # Extraction de documents
│   ├── retriever.py       # Récupération d'information
│   ├── generator.py       # Génération de texte
│   ├── evaluator.py       # Évaluation des résultats
│   └── vectorstore.py     # Gestion de la base vectorielle
├── scripts/
│   ├── ingest.py          # Script d'ingestion
│   └── evaluate.py        # Script d'évaluation
├── tests/                 # Tests unitaires
└── docker-compose.yml     # Configuration Docker
```

## 🛠️ Fonctionnalités principales

- **Extraction multimodale** : PDF, images, tableaux
- **Indexation vectorielle** : Chromadb
- **Récupération intelligente** : LangChain avec embeddings
- **Génération contextuelle** : API OpenAI
- **Évaluation RAGAS** : Qualité des réponses
- **API REST** : FastAPI avec documentation Swagger
- **Tests** : Pytest avec couverture

## 📋 Dépendances principales

- **LangChain** : Framework RAG
- **ChromaDB** : Base de données vectorielle
- **Sentence-Transformers** : Embeddings
- **FastAPI** : API web
- **PyMuPDF & PDFPlumber** : Traitement PDF

## ⚙️ Configuration

Voir [.env.example](.env.example) pour la liste complète des variables d'environnement.

Variables essentielles :
```
OPENAI_API_KEY=votre_cle_api
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
```

##  Tests

```bash
poetry run pytest
poetry run pytest --cov=src tests/  # Avec couverture
```

## 📚 Documentation API

L'API est documentée automatiquement avec Swagger :
```
http://localhost:8000/docs
```

## Déploiement Docker sur Render

Le projet est prêt pour un déploiement conteneurisé via [render.yaml](../render.yaml) et [Dockerfile](Dockerfile).

Étapes:
1. Pousser le repository sur GitHub.
2. Sur Render, créer un service Blueprint (ou importer le repository).
3. Vérifier les variables d'environnement:
	- OPENAI_API_KEY (secret)
	- OPENAI_API_BASE=https://api.openai.com/v1
	- ENABLE_IMAGE_EXTRACTION=true (pour PDF scannés)
4. Déployer. Render injecte automatiquement PORT, utilisé par la commande uvicorn du conteneur.

Test local du conteneur:
1. Depuis le dossier audit_rag, lancer: docker build -t audit-rag-api .
2. Lancer: docker run --rm -p 8000:8000 --env-file .env audit-rag-api
3. Ouvrir: http://localhost:8000/docs

## 🤝 Contribution

Les contributions sont bienvenues ! N'hésitez pas à ouvrir des issues ou des pull requests.

## 📝 Licence

À définir

## 👤 Auteur

Damtoti

---

**Dernière mise à jour :** Mars 2026
