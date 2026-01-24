"""
Advanced RAG Module for Backstory Consistency Verification

Production-grade implementation with <1% hallucination tolerance.
Features:
- Hierarchical chunking with temporal metadata
- Hybrid retrieval (Semantic + BM25 + RRF)
- GraphRAG with event-edge knowledge graphs
- Confidence-based adaptive routing
- Chain-of-Verification generation
- Self-consistency decoding

Usage:
    from advanced_rag import AdvancedRAGPipeline
    
    pipeline = AdvancedRAGPipeline()
    pipeline.ingest_books("./Dataset/Books")
    result = pipeline.verify_backstory(backstory, character, book)
"""

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.pipeline import AdvancedRAGPipeline, create_pipeline

__all__ = [
    "AdvancedRAGConfig",
    "DEFAULT_CONFIG",
    "AdvancedRAGPipeline",
    "create_pipeline",
]

__version__ = "1.0.0"
