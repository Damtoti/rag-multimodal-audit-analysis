"""Génération de réponses RAG pour l'analyse d'audit."""
import logging
from typing import Any, Optional
 
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI
 
from audit_rag.config import Settings, get_settings
from audit_rag.retriever import AuditRetriever
 
logger = logging.getLogger(__name__)
 
SYSTEM_PROMPT = """Tu es un auditeur financier expert. Tu réponds UNIQUEMENT à partir \
du contexte fourni (extraits de textes, descriptions de tableaux et d\'images).
 
Règles :
- Cite toujours la source et le numéro de page.
- Signale explicitement les tableaux et graphiques pertinents.
- Signale toute anomalie ou risque identifié.
- Si l\'information est absente du contexte, indique-le clairement.
- Réponds en français.
 
Contexte :
{context}
 
Question : {question}
 
Réponse :"""
 
 
class AuditRAGGenerator:
    """Génère des réponses structurées à partir des documents récupérés."""
 
    def __init__(
        self,
        retriever: AuditRetriever,
        settings: Optional[Settings] = None,
    ):
        self.retriever = retriever
        self.cfg       = settings or get_settings()

        self.llm = ChatOpenAI(
            model=self.cfg.llm_model,
            temperature=0.1,
            max_tokens=2000,
            api_key=self.cfg.openai_api_key,
        )

        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
 
    def answer(
        self,
        question: str,
        k: int = 6,
        use_mmr: bool = True,
    ) -> dict[str, Any]:
        docs    = self.retriever.retrieve(question, k=k, use_mmr=use_mmr)
        context = self._format_context(docs)

        if not docs:
            return {
                "question":    question,
                "answer":      "Aucun document trouvé pour répondre.",
                "source_docs": [],
                "metadata": {
                    "num_docs": 0,
                    "types": [],
                    "sources": [],
                    "pages": [],
                },
            }

        chain = self.prompt | self.llm
        resp  = chain.invoke({"context": context, "question": question})
        answer_text = resp.content

        return {
            "question":    question,
            "answer":      answer_text,
            "source_docs": docs,
            "metadata": {
                "num_docs":  len(docs),
                "types":     [d.metadata.get("type") for d in docs],
                "sources":   list({d.metadata.get("source") for d in docs}),
                "pages":     sorted({d.metadata.get("page") for d in docs}),
            },
        }
 
    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        parts: list[str] = []
        labels = {"text": "Texte", "table": "Tableau", "image": "Image"}
        for i, doc in enumerate(docs, start=1):
            meta   = doc.metadata
            label  = labels.get(meta.get("type", "text"), "Élément")
            source = meta.get("source", "?")
            page   = meta.get("page", "?")
            parts.append(f"[{label} #{i} | {source} p.{page}]\\n{doc.page_content}")
        return "\\n\\n---\\n\\n".join(parts)