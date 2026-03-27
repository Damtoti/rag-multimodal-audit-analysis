"""Évaluation du système RAG avec RAGAS."""
import logging
from typing import Any
 
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
 
from audit_rag.generator import AuditRAGGenerator
 
logger = logging.getLogger(__name__)
 
 
def run_evaluation(
    generator: AuditRAGGenerator,
    test_questions: list[str],
    ground_truths: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Évalue le système RAG sur un jeu de questions de référence."""
 
    logger.info("Démarrage évaluation RAGAS (%d questions)", len(test_questions))
    eval_data: dict[str, list[Any]] = {
        "question":     [],
        "answer":       [],
        "contexts":     [],
        "ground_truth": [],
    }
 
    for question, gt in zip(test_questions, ground_truths):
        result = generator.answer(question)
        eval_data["question"].append(question)
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(
            [doc.page_content for doc in result["source_docs"]]
        )
        eval_data["ground_truth"].append(gt)
 
    dataset = Dataset.from_dict(eval_data)
    scores  = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    df = scores.to_pandas()
 
    summary = {
        "faithfulness":       df["faithfulness"].mean(),
        "answer_relevancy":   df["answer_relevancy"].mean(),
        "context_precision":  df["context_precision"].mean(),
        "context_recall":     df["context_recall"].mean(),
    }
    logger.info("Résultats RAGAS : %s", summary)
    return df, summary