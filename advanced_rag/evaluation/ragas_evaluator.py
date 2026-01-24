"""
RAGAS Evaluator for RAG Triad Metrics

Integrates RAGAS (Retrieval Augmented Generation Assessment) to measure:
1. Faithfulness: Is the answer grounded in retrieved context?
2. Answer Relevance: Does the answer address the question?
3. Context Precision: Are the retrieved docs relevant?
4. Context Recall: Did we retrieve ALL relevant info?

Target thresholds for production:
- Faithfulness: >0.95 (minimize hallucinations)
- Context Precision: >0.85
- Answer Relevance: >0.90
"""

import logging
from typing import Optional, List, Dict
from dataclasses import dataclass

import pandas as pd

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class RAGASMetrics:
    """Computed RAGAS metrics for a single example."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None


@dataclass
class RAGASEvaluation:
    """Aggregated RAGAS evaluation results."""
    n_samples: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: Optional[float] = None
    mean_answer_correctness: Optional[float] = None
    
    # Distribution stats
    std_faithfulness: float = 0.0
    std_answer_relevancy: float = 0.0
    
    # Threshold compliance
    faithfulness_pass_rate: float = 0.0  # % above 0.95
    precision_pass_rate: float = 0.0     # % above 0.85


class RAGASEvaluator:
    """
    RAGAS evaluation for RAG Triad metrics.
    
    The RAG Triad:
    1. Faithfulness: Is the answer grounded in retrieved context?
       - Penalizes hallucinations
       - Target: >0.95
    
    2. Answer Relevance: Does the answer address the question?
       - Penalizes tangential responses
       - Target: >0.90
    
    3. Context Precision: Are retrieved docs relevant?
       - Measures retrieval quality
       - Target: >0.85
    
    Usage:
        evaluator = RAGASEvaluator()
        metrics = evaluator.evaluate(questions, contexts, answers)
    """
    
    def __init__(
        self,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
        llm_model: str = None,
    ):
        self.config = config
        self.llm_model = llm_model or config.llm.model_name
        
        # Try to import RAGAS
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            self.ragas_available = True
            self.evaluate_fn = evaluate
            self.metrics = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }
        except ImportError:
            logger.warning("RAGAS not installed. Run: pip install ragas")
            self.ragas_available = False
    
    def _prepare_dataset(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None,
    ):
        """Prepare RAGAS-compatible dataset."""
        from datasets import Dataset
        
        data = {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        return Dataset.from_dict(data)
    
    def evaluate(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> RAGASEvaluation:
        """
        Run RAGAS evaluation on a set of examples.
        
        Args:
            questions: List of questions/claims
            contexts: List of retrieved context lists
            answers: List of generated answers/verifications
            ground_truths: Optional ground truth answers
            
        Returns:
            RAGASEvaluation with aggregated metrics
        """
        if not self.ragas_available:
            return self._fallback_evaluation(questions, contexts, answers)
        
        try:
            dataset = self._prepare_dataset(
                questions, contexts, answers, ground_truths
            )
            
            # Select metrics based on whether ground truth is available
            metrics_to_use = [
                self.metrics["faithfulness"],
                self.metrics["answer_relevancy"],
                self.metrics["context_precision"],
            ]
            
            if ground_truths:
                metrics_to_use.append(self.metrics["context_recall"])
            
            # Run evaluation
            result = self.evaluate_fn(dataset, metrics=metrics_to_use)
            
            # Convert to our format
            df = result.to_pandas()
            
            return RAGASEvaluation(
                n_samples=len(df),
                mean_faithfulness=df["faithfulness"].mean(),
                mean_answer_relevancy=df["answer_relevancy"].mean(),
                mean_context_precision=df["context_precision"].mean(),
                mean_context_recall=df["context_recall"].mean() if "context_recall" in df else None,
                std_faithfulness=df["faithfulness"].std(),
                std_answer_relevancy=df["answer_relevancy"].std(),
                faithfulness_pass_rate=(df["faithfulness"] >= 0.95).mean(),
                precision_pass_rate=(df["context_precision"] >= 0.85).mean(),
            )
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return self._fallback_evaluation(questions, contexts, answers)
    
    def _fallback_evaluation(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
    ) -> RAGASEvaluation:
        """
        Simple fallback evaluation when RAGAS is unavailable.
        
        Uses heuristics:
        - Faithfulness: Check if answer tokens appear in context
        - Relevancy: Check question-answer token overlap
        - Precision: Check context-question token overlap
        """
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        
        for q, ctx_list, a in zip(questions, contexts, answers):
            # Simple token overlap heuristics
            q_tokens = set(q.lower().split())
            a_tokens = set(a.lower().split())
            ctx_tokens = set(" ".join(ctx_list).lower().split())
            
            # Faithfulness: answer tokens in context
            if a_tokens:
                faith = len(a_tokens & ctx_tokens) / len(a_tokens)
            else:
                faith = 0.0
            faithfulness_scores.append(faith)
            
            # Relevancy: question-answer overlap
            if q_tokens:
                rel = len(q_tokens & a_tokens) / len(q_tokens)
            else:
                rel = 0.0
            relevancy_scores.append(min(rel * 2, 1.0))  # Scale up
            
            # Precision: context-question overlap
            if q_tokens:
                prec = len(q_tokens & ctx_tokens) / len(q_tokens)
            else:
                prec = 0.0
            precision_scores.append(prec)
        
        return RAGASEvaluation(
            n_samples=len(questions),
            mean_faithfulness=sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            mean_answer_relevancy=sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0,
            mean_context_precision=sum(precision_scores) / len(precision_scores) if precision_scores else 0,
            faithfulness_pass_rate=sum(1 for f in faithfulness_scores if f >= 0.95) / len(faithfulness_scores) if faithfulness_scores else 0,
            precision_pass_rate=sum(1 for p in precision_scores if p >= 0.85) / len(precision_scores) if precision_scores else 0,
        )
    
    def evaluate_pipeline(
        self,
        pipeline,
        test_data: pd.DataFrame,
        limit: int = None,
    ) -> Dict:
        """
        Evaluate a RAG pipeline on test data.
        
        Args:
            pipeline: AdvancedRAGPipeline instance
            test_data: DataFrame with 'content', 'char', 'book_name' columns
            limit: Optional limit on samples
            
        Returns:
            Dict with metrics and per-sample results
        """
        if limit:
            test_data = test_data.head(limit)
        
        questions = []
        contexts = []
        answers = []
        
        from tqdm import tqdm
        
        for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
            backstory = row.get('content') or row.get('backstory')
            character = row.get('char') or row.get('Character')
            book = row.get('book_name') or row.get('Book')
            
            try:
                result = pipeline.verify_backstory(backstory, character, book)
                
                questions.append(backstory)
                contexts.append([result.rationale] if result.rationale else [""])
                answers.append(result.verdict.value)
                
            except Exception as e:
                logger.error(f"Error: {e}")
                questions.append(backstory)
                contexts.append([""])
                answers.append("ERROR")
        
        evaluation = self.evaluate(questions, contexts, answers)
        
        return {
            "ragas_metrics": evaluation,
            "n_samples": len(questions),
        }
    
    def print_report(self, evaluation: RAGASEvaluation) -> None:
        """Print formatted evaluation report."""
        print("\n" + "="*50)
        print("RAGAS EVALUATION REPORT")
        print("="*50)
        print(f"Samples: {evaluation.n_samples}")
        print("-"*50)
        print(f"Faithfulness:       {evaluation.mean_faithfulness:.3f} (target: >0.95)")
        print(f"Answer Relevancy:   {evaluation.mean_answer_relevancy:.3f} (target: >0.90)")
        print(f"Context Precision:  {evaluation.mean_context_precision:.3f} (target: >0.85)")
        if evaluation.mean_context_recall:
            print(f"Context Recall:     {evaluation.mean_context_recall:.3f}")
        print("-"*50)
        print(f"Faithfulness Pass Rate (>0.95): {evaluation.faithfulness_pass_rate:.1%}")
        print(f"Precision Pass Rate (>0.85):    {evaluation.precision_pass_rate:.1%}")
        print("="*50 + "\n")
