"""
BM25 Indexer for Sparse Keyword Search

Provides exact-match keyword retrieval to complement semantic search.
Essential for entity names, dates, and specific terms that embedding
models may not precisely match.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
import re

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class BM25SearchResult:
    """Result from BM25 search."""
    chunk_id: str
    text: str
    score: float
    rank: int


class BM25Indexer:
    """
    BM25 sparse index for keyword-based retrieval.
    
    Complements dense semantic search by:
    - Exact matching of entity names
    - Handling rare/specific terms
    - Boosting explicitly mentioned keywords
    
    BM25 (Best Match 25) is a TF-IDF variant that:
    - Rewards term frequency with saturation
    - Penalizes long documents
    - Uses tunable k1 and b parameters
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
    ):
        self.k1 = k1
        self.b = b
        self.config = config
        
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.texts: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Simple whitespace + lowercasing.
        Could be enhanced with stemming/lemmatization.
        """
        # Remove special characters, lowercase, split
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_index(
        self,
        chunks: List[dict],
        show_progress: bool = False,
    ) -> None:
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of dicts with 'chunk_id' and 'text'
            show_progress: Whether to show progress (not used, for API compat)
        """
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.texts = [c["text"] for c in chunks]
        
        logger.info(f"Building BM25 index from {len(chunks)} chunks...")
        
        self.tokenized_corpus = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        
        logger.info(f"BM25 index built with {len(self.chunk_ids)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[BM25SearchResult]:
        """
        Search the BM25 index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum score to include
            
        Returns:
            List of BM25SearchResult sorted by score
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        tokenized_query = self._tokenize(query)
        
        if not tokenized_query:
            return []
        
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            score = scores[idx]
            if score < score_threshold:
                continue
            
            results.append(BM25SearchResult(
                chunk_id=self.chunk_ids[idx],
                text=self.texts[idx],
                score=float(score),
                rank=rank + 1,
            ))
        
        return results
    
    def get_document_frequency(self, term: str) -> int:
        """Get number of documents containing a term."""
        term_lower = term.lower()
        count = 0
        for doc in self.tokenized_corpus:
            if term_lower in doc:
                count += 1
        return count
    
    def get_idf(self, term: str) -> float:
        """Get IDF score for a term."""
        if self.bm25 is None:
            return 0.0
        
        term_lower = term.lower()
        if term_lower in self.bm25.idf:
            return self.bm25.idf[term_lower]
        return 0.0
