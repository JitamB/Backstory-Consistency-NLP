"""
Hybrid Retriever with Semantic + BM25 + RRF Fusion + Temporal Decay

Implements production-grade retrieval combining:
1. Dense semantic search (BGE embeddings)
2. Sparse keyword search (BM25)
3. Reciprocal Rank Fusion for score combination
4. Temporal decay based on query intent

Addresses the problem where semantic search alone misses exact entity matches
and BM25 alone misses semantic similarities.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.generation.schemas import QueryIntent, ChunkMetadata

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval with all scores."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    
    # Individual scores
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    temporal_score: float = 0.0
    
    # Rank positions (lower is better)
    semantic_rank: int = 0
    bm25_rank: int = 0
    
    # Combined scores
    rrf_score: float = 0.0
    final_score: float = 0.0
    
    # Metadata
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None


@dataclass
class HybridSearchIndex:
    """Index for hybrid search containing both dense and sparse indices."""
    chunk_ids: list[str]
    texts: list[str]
    metadata: list[ChunkMetadata]
    
    # Dense vectors
    embeddings: np.ndarray = None
    
    # Sparse index
    bm25: BM25Okapi = None
    tokenized_corpus: list[list[str]] = field(default_factory=list)
    
    # Parent mapping for Small-to-Big retrieval
    child_to_parent: dict[str, str] = field(default_factory=dict)
    parent_texts: dict[str, str] = field(default_factory=dict)


class HybridRetriever:
    """
    Production-grade hybrid retriever combining semantic and keyword search.
    
    Features:
    - Semantic search with BGE/E5 embeddings
    - BM25 keyword search for exact matches
    - Reciprocal Rank Fusion (RRF) for score combination
    - Temporal decay based on query intent
    - Small-to-Big retrieval (search children, return parents)
    """
    
    def __init__(
        self,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
        embedding_model: Optional[SentenceTransformer] = None,
    ):
        self.config = config
        
        # Initialize embedding model
        if embedding_model:
            self.embedder = embedding_model
        else:
            logger.info(f"Loading embedding model: {config.embedding.model_name}")
            self.embedder = SentenceTransformer(config.embedding.model_name)
        
        self.index: Optional[HybridSearchIndex] = None
        
    def build_index(
        self,
        chunks: list[dict],
        show_progress: bool = True,
    ) -> HybridSearchIndex:
        """
        Build hybrid search index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_id', 'text', 'metadata'
            show_progress: Whether to show progress bar
            
        Returns:
            HybridSearchIndex ready for search
        """
        logger.info(f"Building hybrid index from {len(chunks)} chunks")
        
        # Extract data
        chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadata = [c["metadata"] if isinstance(c["metadata"], ChunkMetadata) 
                   else ChunkMetadata(**c["metadata"]) for c in chunks]
        
        # Build parent mapping for Small-to-Big
        child_to_parent = {}
        parent_texts = {}
        for c in chunks:
            if c.get("parent_id"):
                child_to_parent[c["chunk_id"]] = c["parent_id"]
            if c.get("is_parent", False) or not c.get("parent_id"):
                parent_texts[c["chunk_id"]] = c["text"]
        
        # Create dense embeddings
        logger.info("Creating dense embeddings...")
        embeddings = self.embedder.encode(
            texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.embedding.normalize,
            convert_to_numpy=True,
        )
        
        # Create BM25 index
        logger.info("Creating BM25 index...")
        tokenized_corpus = [self._tokenize(text) for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        self.index = HybridSearchIndex(
            chunk_ids=chunk_ids,
            texts=texts,
            metadata=metadata,
            embeddings=embeddings,
            bm25=bm25,
            tokenized_corpus=tokenized_corpus,
            child_to_parent=child_to_parent,
            parent_texts=parent_texts,
        )
        
        logger.info(f"Index built: {len(chunk_ids)} chunks, "
                   f"embedding dim={embeddings.shape[1]}")
        
        return self.index
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _calculate_rrf(
        self,
        semantic_ranks: dict[str, int],
        bm25_ranks: dict[str, int],
        k: int = 60,
    ) -> dict[str, float]:
        """
        Calculate Reciprocal Rank Fusion scores.
        
        RRF_score(d) = Σ 1/(k + rank_i(d)) for each retriever i
        
        Args:
            semantic_ranks: {chunk_id: rank} from semantic search
            bm25_ranks: {chunk_id: rank} from BM25 search
            k: RRF constant (default 60)
            
        Returns:
            {chunk_id: rrf_score}
        """
        all_docs = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores = {}
        
        # Get weights from config
        semantic_weight = self.config.retrieval.semantic_weight
        bm25_weight = self.config.retrieval.bm25_weight
        
        for doc_id in all_docs:
            score = 0.0
            
            # Semantic contribution
            if doc_id in semantic_ranks:
                score += semantic_weight * (1.0 / (k + semantic_ranks[doc_id]))
            
            # BM25 contribution
            if doc_id in bm25_ranks:
                score += bm25_weight * (1.0 / (k + bm25_ranks[doc_id]))
            
            rrf_scores[doc_id] = score
        
        return rrf_scores
    
    def _apply_temporal_decay(
        self,
        results: list[RetrievalResult],
        query_intent: QueryIntent,
        decay_strength: Optional[float] = None,
    ) -> list[RetrievalResult]:
        """
        Apply temporal decay based on query intent.
        
        - CURRENT_STATE: Boost chunks from later in narrative
        - ORIGIN_STORY: Boost chunks from earlier in narrative
        - GENERAL: No temporal bias
        
        Args:
            results: List of retrieval results
            query_intent: Classified query intent
            decay_strength: Override for decay strength (0-1)
            
        Returns:
            Results with updated temporal_score and final_score
        """
        strength = decay_strength or self.config.retrieval.temporal_decay_strength
        
        for result in results:
            narrative_pos = result.metadata.narrative_position
            
            if query_intent == QueryIntent.CURRENT_STATE:
                # Boost later chunks: score increases with position
                result.temporal_score = narrative_pos * strength
            elif query_intent == QueryIntent.ORIGIN_STORY:
                # Boost earlier chunks: score decreases with position
                result.temporal_score = (1.0 - narrative_pos) * strength
            else:
                # No temporal bias
                result.temporal_score = 0.0
            
            # Update final score
            result.final_score = result.rrf_score + result.temporal_score
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = None,
        book_filter: Optional[str] = None,
        query_intent: QueryIntent = QueryIntent.GENERAL,
        return_parents: bool = True,
    ) -> list[RetrievalResult]:
        """
        Perform hybrid search with RRF fusion and temporal decay.
        
        Args:
            query: Search query
            top_k: Number of results to return (default from config)
            book_filter: Optional book name to filter by
            query_intent: Query intent for temporal decay
            return_parents: Whether to return parent chunks (Small-to-Big)
            
        Returns:
            List of RetrievalResult sorted by final_score
        """
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")
        
        top_k = top_k or self.config.retrieval.initial_candidates
        
        # Apply book filter if specified
        if book_filter:
            valid_indices = [
                i for i, m in enumerate(self.index.metadata)
                if book_filter.lower() in str(getattr(m, 'book_name', '')).lower()
            ]
            if not valid_indices:
                logger.warning(f"No chunks found for book: {book_filter}")
                return []
        else:
            valid_indices = list(range(len(self.index.chunk_ids)))
        
        if not valid_indices:
            return []
        
        # === Semantic Search ===
        query_embedding = self.embedder.encode(
            query,
            normalize_embeddings=self.config.embedding.normalize,
            convert_to_numpy=True,
        )
        
        # Filter embeddings
        filtered_embeddings = self.index.embeddings[valid_indices]
        
        # Calculate similarities
        similarities = util.cos_sim(query_embedding, filtered_embeddings)[0].numpy()
        
        # Get top-K semantic results
        semantic_top_k = min(top_k, len(valid_indices))
        semantic_top_indices = np.argsort(similarities)[::-1][:semantic_top_k]
        
        semantic_ranks = {}
        semantic_scores = {}
        for rank, idx in enumerate(semantic_top_indices):
            original_idx = valid_indices[idx]
            chunk_id = self.index.chunk_ids[original_idx]
            semantic_ranks[chunk_id] = rank + 1  # 1-indexed
            semantic_scores[chunk_id] = float(similarities[idx])
        
        # === BM25 Search ===
        tokenized_query = self._tokenize(query)
        
        # Get all BM25 scores
        bm25_scores_all = self.index.bm25.get_scores(tokenized_query)
        
        # Filter to valid indices
        bm25_scores_filtered = [(valid_indices[i], bm25_scores_all[valid_indices[i]]) 
                                for i in range(len(valid_indices))]
        bm25_scores_filtered.sort(key=lambda x: x[1], reverse=True)
        
        bm25_ranks = {}
        bm25_scores = {}
        for rank, (original_idx, score) in enumerate(bm25_scores_filtered[:top_k]):
            chunk_id = self.index.chunk_ids[original_idx]
            bm25_ranks[chunk_id] = rank + 1  # 1-indexed
            bm25_scores[chunk_id] = float(score)
        
        # === RRF Fusion ===
        rrf_scores = self._calculate_rrf(
            semantic_ranks, 
            bm25_ranks, 
            k=self.config.retrieval.rrf_k
        )
        
        # === Build Results ===
        results = []
        all_chunk_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
        
        for chunk_id in all_chunk_ids:
            idx = self.index.chunk_ids.index(chunk_id)
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                text=self.index.texts[idx],
                metadata=self.index.metadata[idx],
                semantic_score=semantic_scores.get(chunk_id, 0.0),
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                semantic_rank=semantic_ranks.get(chunk_id, 999),
                bm25_rank=bm25_ranks.get(chunk_id, 999),
                rrf_score=rrf_scores.get(chunk_id, 0.0),
                final_score=rrf_scores.get(chunk_id, 0.0),
            )
            
            # Add parent text for Small-to-Big retrieval
            if return_parents:
                parent_id = self.index.child_to_parent.get(chunk_id)
                if parent_id:
                    result.parent_id = parent_id
                    result.parent_text = self.index.parent_texts.get(parent_id)
            
            results.append(result)
        
        # === Apply Temporal Decay ===
        results = self._apply_temporal_decay(results, query_intent)
        
        # Sort by final score and return top-k
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results[:top_k]
    
    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = None,
        book_filter: Optional[str] = None,
        query_intent: QueryIntent = QueryIntent.GENERAL,
        aggregation: str = "max",  # "max", "mean", "rrf"
    ) -> list[RetrievalResult]:
        """
        Search with multiple query variants and aggregate results.
        
        Args:
            queries: List of query variants (from HyDE/Multi-Query)
            top_k: Number of final results
            book_filter: Optional book filter
            query_intent: Query intent for temporal decay
            aggregation: How to combine scores ("max", "mean", "rrf")
            
        Returns:
            Aggregated list of RetrievalResult
        """
        all_results: dict[str, list[RetrievalResult]] = defaultdict(list)
        
        # Search with each query
        for query in queries:
            results = self.search(
                query, 
                top_k=top_k or self.config.retrieval.initial_candidates,
                book_filter=book_filter,
                query_intent=query_intent,
            )
            for r in results:
                all_results[r.chunk_id].append(r)
        
        # Aggregate results
        aggregated = []
        for chunk_id, result_list in all_results.items():
            base_result = result_list[0]  # Use first result as base
            
            if aggregation == "max":
                base_result.final_score = max(r.final_score for r in result_list)
            elif aggregation == "mean":
                base_result.final_score = np.mean([r.final_score for r in result_list])
            elif aggregation == "rrf":
                # Apply RRF across queries
                rrf_sum = 0
                for i, r in enumerate(sorted(result_list, key=lambda x: x.final_score, reverse=True)):
                    rrf_sum += 1.0 / (60 + i + 1)
                base_result.final_score = rrf_sum
            
            aggregated.append(base_result)
        
        # Sort and return
        aggregated.sort(key=lambda x: x.final_score, reverse=True)
        return aggregated[:top_k or self.config.retrieval.initial_candidates]
    
    def get_parent_context(self, result: RetrievalResult) -> Optional[str]:
        """Get parent chunk text for context expansion."""
        if result.parent_text:
            return result.parent_text
        if result.parent_id and self.index:
            return self.index.parent_texts.get(result.parent_id)
        return None


class TemporalDecayCalculator:
    """
    Utility class for calculating temporal decay scores.
    
    Separates decay logic for testing and reuse.
    """
    
    @staticmethod
    def linear_decay(position: float, intent: QueryIntent, strength: float = 0.3) -> float:
        """Linear decay based on narrative position."""
        if intent == QueryIntent.CURRENT_STATE:
            return position * strength
        elif intent == QueryIntent.ORIGIN_STORY:
            return (1.0 - position) * strength
        return 0.0
    
    @staticmethod
    def exponential_decay(position: float, intent: QueryIntent, strength: float = 0.3) -> float:
        """Exponential decay for stronger position bias."""
        if intent == QueryIntent.CURRENT_STATE:
            return (1 - np.exp(-3 * position)) * strength
        elif intent == QueryIntent.ORIGIN_STORY:
            return (1 - np.exp(-3 * (1 - position))) * strength
        return 0.0
    
    @staticmethod
    def chapter_boost(
        chapter_index: int,
        total_chapters: int,
        intent: QueryIntent,
        boost: float = 0.2,
    ) -> float:
        """Boost based on chapter position."""
        if total_chapters <= 1:
            return 0.0
        
        chapter_position = chapter_index / (total_chapters - 1)
        
        if intent == QueryIntent.CURRENT_STATE:
            # Boost last few chapters
            return boost if chapter_index >= total_chapters - 2 else 0.0
        elif intent == QueryIntent.ORIGIN_STORY:
            # Boost first few chapters
            return boost if chapter_index <= 1 else 0.0
        return 0.0
