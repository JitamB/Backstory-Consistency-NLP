"""
Advanced Cross-Encoder Reranker with BGE-Reranker-v2

Features:
- BGE-Reranker-v2-M3 for high-quality reranking
- Recency-weighted scoring based on query intent
- Score thresholding for noise filtering
- Contextual compression for long chunks

Upgrades from ms-marco-MiniLM (62.1% MTEB) to BGE-Reranker (67.5% MTEB).
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

import numpy as np

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.generation.schemas import QueryIntent
from advanced_rag.retrieval.hybrid_retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result after reranking with updated scores."""
    original_result: RetrievalResult
    rerank_score: float
    recency_adjustment: float
    final_rerank_score: float
    passed_threshold: bool


class AdvancedReranker:
    """
    High-quality cross-encoder reranker.
    
    Features:
    - BGE-Reranker-v2-M3 (supports up to 8192 tokens)
    - Recency-weighted reranking based on query intent
    - Score threshold filtering
    - Batch processing for efficiency
    """
    
    def __init__(
        self,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
        model=None,
    ):
        self.config = config
        
        # Load reranker model
        if model:
            self.model = model
        else:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading reranker: {config.reranker.model_name}")
                self.model = CrossEncoder(
                    config.reranker.model_name,
                    max_length=config.reranker.max_length,
                )
            except Exception as e:
                logger.warning(f"Failed to load BGE-Reranker: {e}")
                logger.info("Falling back to ms-marco-MiniLM")
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def _calculate_recency_adjustment(
        self,
        narrative_position: float,
        query_intent: QueryIntent,
        strength: float = 0.2,
    ) -> float:
        """
        Calculate recency adjustment based on narrative position and intent.
        
        - CURRENT_STATE: Boost later passages
        - ORIGIN_STORY: Boost earlier passages
        - GENERAL: No adjustment
        """
        if query_intent == QueryIntent.CURRENT_STATE:
            return narrative_position * strength
        elif query_intent == QueryIntent.ORIGIN_STORY:
            return (1.0 - narrative_position) * strength
        return 0.0
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None,
        query_intent: QueryIntent = QueryIntent.GENERAL,
        apply_threshold: bool = True,
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Search query
            results: Initial retrieval results
            top_k: Number of results to return
            query_intent: Intent for recency adjustment
            apply_threshold: Whether to filter by score threshold
            
        Returns:
            Reranked list of RetrievalResult
        """
        if not results:
            return []
        
        top_k = top_k or self.config.retrieval.after_rerank
        
        # Prepare pairs for cross-encoder
        pairs = [(query, r.text) for r in results]
        
        # Batch score
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.config.reranker.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:top_k]
        
        # Convert to numpy for operations
        scores = np.array(scores)
        
        # Normalize scores to 0-1 if needed
        if scores.min() < 0 or scores.max() > 1:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        # Apply recency adjustment
        for i, (result, score) in enumerate(zip(results, scores)):
            recency = self._calculate_recency_adjustment(
                result.metadata.narrative_position,
                query_intent,
            )
            scores[i] = score + recency
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        
        # Apply threshold filter
        threshold = self.config.reranker.score_threshold
        reranked = []
        
        for idx in sorted_indices:
            if apply_threshold and scores[idx] < threshold:
                continue
            
            result = results[idx]
            # Update final score with rerank score
            result.final_score = float(scores[idx])
            reranked.append(result)
            
            if len(reranked) >= top_k:
                break
        
        logger.info(f"Reranked {len(results)} -> {len(reranked)} results "
                   f"(threshold={threshold}, intent={query_intent.value})")
        
        return reranked
    
    def rerank_with_parent_context(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None,
        query_intent: QueryIntent = QueryIntent.GENERAL,
    ) -> List[RetrievalResult]:
        """
        Rerank using parent context for better scoring.
        
        For Small-to-Big retrieval: scores against parent chunk text.
        """
        if not results:
            return []
        
        # Use parent text if available, otherwise use chunk text
        pairs = []
        for r in results:
            context = r.parent_text if r.parent_text else r.text
            pairs.append((query, context))
        
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.config.reranker.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Parent context reranking failed: {e}")
            return self.rerank(query, results, top_k, query_intent)
        
        scores = np.array(scores)
        if scores.min() < 0 or scores.max() > 1:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        # Apply recency
        for i, result in enumerate(results):
            recency = self._calculate_recency_adjustment(
                result.metadata.narrative_position,
                query_intent,
            )
            scores[i] += recency
        
        # Sort and return
        sorted_indices = np.argsort(scores)[::-1]
        top_k = top_k or self.config.retrieval.after_rerank
        
        reranked = []
        for idx in sorted_indices[:top_k]:
            result = results[idx]
            result.final_score = float(scores[idx])
            reranked.append(result)
        
        return reranked
