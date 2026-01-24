"""
Centralized configuration for Advanced RAG system.
All configurable parameters in one place.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "BAAI/bge-large-en-v1.5"
    dimension: int = 1024
    batch_size: int = 32

    # Using faster model for CPU - switch to BGE for GPU
    # model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # 30x faster than BGE-large
    # dimension: int = 384
    # batch_size: int = 64  # Larger batch for faster processing
    normalize: bool = True


@dataclass
class ChunkingConfig:
    """Hierarchical chunking configuration."""
    # Parent chunks (for context)
    parent_chunk_size: int = 2048
    parent_overlap: int = 256
    
    # Child chunks (for precision matching)
    child_chunk_size: int = 256
    child_overlap: int = 32
    
    # Chapter detection
    chapter_patterns: list[str] = field(default_factory=lambda: [
        r'CHAPTER\s+(\d+|[IVXLC]+)',
        r'Chapter\s+(\d+|[A-Z][a-z]+)',
        r'^([IVXLC]+)\.\s',
        r'PART\s+(\d+|[IVXLC]+)',
    ])


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    # Hybrid search weights
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4
    
    # RRF parameters
    rrf_k: int = 60
    
    # Top-K at each stage
    initial_candidates: int = 50
    after_rerank: int = 5
    
    # Temporal decay
    temporal_decay_strength: float = 0.3


@dataclass
class RoutingConfig:
    """Confidence-based routing thresholds."""
    fast_threshold: float = 0.85      # Skip HyDE & agent loop
    standard_threshold: float = 0.6   # Use HyDE, skip agent
    # Below standard_threshold: Full pipeline with agentic loop
    
    # Cache settings
    cache_similarity_threshold: float = 0.85
    cache_ttl_hours: int = 24


@dataclass
class RerankerConfig:
    """Reranker model configuration."""
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 8192
    batch_size: int = 16
    score_threshold: float = 0.5


@dataclass
class LLMConfig:
    """LLM configuration for generation and verification."""
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 4096
    
    # Self-consistency
    self_consistency_samples: int = 3
    consistency_temperature: float = 0.3
    
    # Agentic loop
    max_agent_iterations: int = 3


@dataclass
class GraphConfig:
    """Knowledge graph configuration."""
    # Entity extraction
    extract_entities: bool = True
    entity_model: str = "en_core_web_sm"  # spaCy model
    
    # Event extraction
    extract_events: bool = True
    events_per_chunk: int = 5
    
    # Graph traversal
    max_hop_distance: int = 2


@dataclass
class AdvancedRAGConfig:
    """Master configuration combining all sub-configs."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    
    # Global settings
    data_dir: str = "./Dataset/Books"
    output_dir: str = "./generated_files"
    use_gpu: bool = True
    verbose: bool = True
    
    @classmethod
    def from_env(cls) -> "AdvancedRAGConfig":
        """Create config from environment variables with defaults."""
        config = cls()
        
        # Override with environment variables if present
        if model := os.getenv("EMBEDDING_MODEL"):
            config.embedding.model_name = model
        if model := os.getenv("RERANKER_MODEL"):
            config.reranker.model_name = model
        if model := os.getenv("LLM_MODEL"):
            config.llm.model_name = model
        if data_dir := os.getenv("DATA_DIR"):
            config.data_dir = data_dir
            
        return config


# Default configuration instance
DEFAULT_CONFIG = AdvancedRAGConfig()
