"""
Hierarchical Indexer for Small-to-Big Retrieval

Manages parent-child chunk relationships:
- Parent chunks (2048 tokens): Rich context for LLM
- Child chunks (256 tokens): Precise matching

Retrieval strategy: Search on children, return parent context.
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import pandas as pd

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.indexing.graph_chunker import ProcessedChunk

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalIndex:
    """Index structure for hierarchical chunks."""
    parent_chunks: Dict[str, ProcessedChunk] = field(default_factory=dict)
    child_chunks: Dict[str, ProcessedChunk] = field(default_factory=dict)
    child_to_parent: Dict[str, str] = field(default_factory=dict)
    parent_to_children: Dict[str, List[str]] = field(default_factory=dict)


class HierarchicalIndexer:
    """
    Manages hierarchical parent-child chunk relationships.
    
    Small-to-Big Retrieval Pattern:
    1. Search is performed on child chunks (more discriminative)
    2. Parent chunks are returned (more context for LLM)
    
    This solves "lost in the middle" by:
    - Matching specific claims in children
    - Providing surrounding context from parents
    """
    
    def __init__(self, config: AdvancedRAGConfig = DEFAULT_CONFIG):
        self.config = config
        self.index = HierarchicalIndex()
    
    def build_index(self, chunks: List[ProcessedChunk]) -> HierarchicalIndex:
        """
        Build hierarchical index from processed chunks.
        
        Args:
            chunks: List of ProcessedChunk objects
            
        Returns:
            HierarchicalIndex with parent-child mappings
        """
        for chunk in chunks:
            if chunk.parent_id is None:
                # This is a parent chunk
                self.index.parent_chunks[chunk.chunk_id] = chunk
                self.index.parent_to_children[chunk.chunk_id] = chunk.child_ids
            else:
                # This is a child chunk
                self.index.child_chunks[chunk.chunk_id] = chunk
                self.index.child_to_parent[chunk.chunk_id] = chunk.parent_id
        
        logger.info(f"Built hierarchical index: "
                   f"{len(self.index.parent_chunks)} parents, "
                   f"{len(self.index.child_chunks)} children")
        
        return self.index
    
    def get_parent(self, chunk_id: str) -> Optional[ProcessedChunk]:
        """Get parent chunk for a given chunk ID."""
        if chunk_id in self.index.parent_chunks:
            return self.index.parent_chunks[chunk_id]
        
        parent_id = self.index.child_to_parent.get(chunk_id)
        if parent_id:
            return self.index.parent_chunks.get(parent_id)
        
        return None
    
    def get_children(self, parent_id: str) -> List[ProcessedChunk]:
        """Get all children of a parent chunk."""
        child_ids = self.index.parent_to_children.get(parent_id, [])
        return [self.index.child_chunks[cid] 
                for cid in child_ids 
                if cid in self.index.child_chunks]
    
    def get_parent_text(self, chunk_id: str) -> Optional[str]:
        """Get parent chunk text for context expansion."""
        parent = self.get_parent(chunk_id)
        return parent.text if parent else None
    
    def expand_to_parent(
        self,
        child_chunk_ids: List[str],
        deduplicate: bool = True,
    ) -> List[ProcessedChunk]:
        """
        Expand child chunk IDs to their parent chunks.
        
        Args:
            child_chunk_ids: List of child chunk IDs
            deduplicate: Whether to remove duplicate parents
            
        Returns:
            List of parent ProcessedChunk objects
        """
        parents = []
        seen_parents = set()
        
        for child_id in child_chunk_ids:
            parent = self.get_parent(child_id)
            if parent:
                if deduplicate and parent.chunk_id in seen_parents:
                    continue
                seen_parents.add(parent.chunk_id)
                parents.append(parent)
        
        return parents
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert index to DataFrame for inspection."""
        records = []
        
        for chunk_id, chunk in self.index.parent_chunks.items():
            records.append({
                "chunk_id": chunk_id,
                "is_parent": True,
                "parent_id": None,
                "children_count": len(chunk.child_ids),
                "word_count": chunk.metadata.word_count,
                "narrative_position": chunk.metadata.narrative_position,
            })
        
        for chunk_id, chunk in self.index.child_chunks.items():
            records.append({
                "chunk_id": chunk_id,
                "is_parent": False,
                "parent_id": chunk.parent_id,
                "children_count": 0,
                "word_count": chunk.metadata.word_count,
                "narrative_position": chunk.metadata.narrative_position,
            })
        
        return pd.DataFrame(records)
