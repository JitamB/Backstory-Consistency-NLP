"""
Graph-Aware Chunker with Temporal Metadata

Implements hierarchical chunking with:
1. Parent-Child chunk relationships (Small-to-Big retrieval)
2. Temporal metadata (chapter index, narrative position)
3. Entity and Event extraction for GraphRAG
4. Scene/location detection

Solves the "narrative time" problem where facts change over the story.
"""

import re
import hashlib
import logging
from typing import Optional, Generator
from dataclasses import dataclass, field

import spacy
from spacy.tokens import Doc
import networkx as nx

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.generation.schemas import (
    ChunkMetadata,
    EntityNode,
    EventEdge,
    EntityExtractionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """A processed chunk with text, metadata, and extracted entities/events."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    entities: list[EntityNode] = field(default_factory=list)
    events: list[EventEdge] = field(default_factory=list)
    
    # Hierarchical relationship
    parent_id: Optional[str] = None
    child_ids: list[str] = field(default_factory=list)


class GraphAwareChunker:
    """
    Advanced chunker with temporal and entity awareness.
    
    Features:
    - Hierarchical parent-child chunking
    - Chapter boundary detection
    - Narrative position tracking (0.0 to 1.0)
    - Entity extraction (characters, locations)
    - Event extraction (who-did-what-to-whom)
    """
    
    # Time context patterns
    TIME_PATTERNS = [
        (r'\b(that night|at night|nightfall|midnight)\b', 'night'),
        (r'\b(morning|sunrise|dawn|daybreak)\b', 'morning'),
        (r'\b(afternoon|midday|noon)\b', 'afternoon'),
        (r'\b(evening|sunset|dusk)\b', 'evening'),
        (r'\b(years? later|months? later|weeks? later)\b', 'time_skip'),
        (r'\b(earlier that day|before|previously|in the past)\b', 'flashback'),
        (r'\b(the next day|tomorrow|the following)\b', 'next_day'),
    ]
    
    # Dialogue detection
    DIALOGUE_PATTERN = re.compile(r'["\'].*?["\']', re.DOTALL)
    
    def __init__(
        self,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
        llm_client=None,  # For event extraction LLM calls
    ):
        self.config = config
        self.llm_client = llm_client
        
        # Load spaCy for entity extraction
        try:
            self.nlp = spacy.load(config.graph.entity_model)
        except OSError:
            logger.warning(f"spaCy model {config.graph.entity_model} not found. "
                         "Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Compile chapter patterns
        self.chapter_patterns = [
            re.compile(p, re.MULTILINE | re.IGNORECASE)
            for p in config.chunking.chapter_patterns
        ]
        
        # Knowledge graph
        self.entity_graph = nx.DiGraph()
        
    def _generate_chunk_id(self, text: str, position: float) -> str:
        """Generate unique chunk ID from content hash and position."""
        content = f"{text[:100]}_{position:.4f}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _detect_chapters(self, text: str) -> list[tuple[int, str, int]]:
        """
        Detect chapter boundaries in text.
        
        Returns:
            List of (chapter_index, chapter_title, start_position)
        """
        chapters = []
        
        for pattern in self.chapter_patterns:
            for match in pattern.finditer(text):
                chapter_num = match.group(1) if match.groups() else str(len(chapters) + 1)
                # Get the full line as chapter title
                line_start = text.rfind('\n', 0, match.start()) + 1
                line_end = text.find('\n', match.end())
                if line_end == -1:
                    line_end = len(text)
                chapter_title = text[line_start:line_end].strip()
                
                chapters.append((len(chapters), chapter_title, match.start()))
        
        # Sort by position and re-index
        chapters.sort(key=lambda x: x[2])
        chapters = [(i, title, pos) for i, (_, title, pos) in enumerate(chapters)]
        
        # If no chapters found, treat entire text as chapter 0
        if not chapters:
            chapters = [(0, "Chapter 1", 0)]
            
        return chapters
    
    def _detect_time_context(self, text: str) -> Optional[str]:
        """Detect temporal context from text patterns."""
        text_lower = text.lower()
        for pattern, context in self.TIME_PATTERNS:
            if re.search(pattern, text_lower):
                return context
        return None
    
    def _is_primarily_dialogue(self, text: str) -> bool:
        """Check if chunk is primarily dialogue."""
        dialogue_matches = self.DIALOGUE_PATTERN.findall(text)
        dialogue_chars = sum(len(m) for m in dialogue_matches)
        return dialogue_chars > len(text) * 0.5
    
    def _extract_entities_spacy(self, doc: Doc) -> list[EntityNode]:
        """Extract entities using spaCy NER."""
        entities = []
        seen = set()
        
        for ent in doc.ents:
            if ent.text.lower() in seen:
                continue
            seen.add(ent.text.lower())
            
            # Map spaCy entity types to our types
            type_map = {
                "PERSON": "CHARACTER",
                "GPE": "LOCATION",
                "LOC": "LOCATION",
                "FAC": "LOCATION",
                "ORG": "ORGANIZATION",
                "PRODUCT": "OBJECT",
            }
            
            entity_type = type_map.get(ent.label_, None)
            if entity_type:
                entities.append(EntityNode(
                    name=ent.text,
                    entity_type=entity_type,
                    aliases=[],
                    first_appearance=0.0,  # Will be updated later
                ))
        
        return entities
    
    def _extract_events_rule_based(
        self,
        doc: Doc,
        entities: list[EntityNode],
        chunk_id: str,
        narrative_position: float,
        chapter_index: int,
    ) -> list[EventEdge]:
        """
        Extract events using dependency parsing.
        
        Looks for Subject-Verb-Object patterns where subject/object are entities.
        """
        events = []
        entity_names = {e.name.lower() for e in entities}
        
        for sent in doc.sents:
            root = sent.root
            
            # Find subject and object
            subject = None
            obj = None
            
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and token.text.lower() in entity_names:
                    subject = token.text
                elif token.dep_ in ("dobj", "pobj", "attr") and token.text.lower() in entity_names:
                    obj = token.text
            
            if subject and obj and root.pos_ == "VERB":
                events.append(EventEdge(
                    subject=subject,
                    action=root.lemma_,
                    object=obj,
                    narrative_position=narrative_position,
                    chapter_index=chapter_index,
                    source_chunk_id=chunk_id,
                    confidence=0.7,  # Rule-based has lower confidence
                ))
        
        return events[:self.config.graph.events_per_chunk]
    
    def _extract_location(self, doc: Doc) -> Optional[str]:
        """Extract scene location from text."""
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                return ent.text
        return None
    
    def _create_child_chunks(
        self,
        parent_text: str,
        parent_id: str,
        parent_metadata: ChunkMetadata,
    ) -> Generator[ProcessedChunk, None, None]:
        """Create child chunks from a parent chunk."""
        words = parent_text.split()
        chunk_size = self.config.chunking.child_chunk_size
        overlap = self.config.chunking.child_overlap
        step = chunk_size - overlap
        
        n_words = len(words)
        child_count = 0
        
        for i in range(0, n_words, step):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < chunk_size // 4:  # Skip very small final chunks
                continue
                
            chunk_text = " ".join(chunk_words)
            
            # Calculate position within parent
            relative_pos = i / n_words if n_words > 0 else 0
            
            # Create child ID
            child_id = f"{parent_id}_c{child_count}"
            child_count += 1
            
            # Child inherits parent metadata with minor adjustments
            # Clamp positions to valid range [0, 1]
            child_narrative_pos = min(1.0, parent_metadata.narrative_position + (relative_pos * 0.01))
            child_chapter_pos = min(1.0, parent_metadata.chapter_position + (relative_pos * 0.01))
            
            child_metadata = ChunkMetadata(
                chunk_id=child_id,
                parent_id=parent_id,
                chapter_index=parent_metadata.chapter_index,
                chapter_title=parent_metadata.chapter_title,
                narrative_position=child_narrative_pos,
                chapter_position=child_chapter_pos,
                scene_location=parent_metadata.scene_location,
                time_context=parent_metadata.time_context,
                entities_mentioned=[],  # Will be populated
                is_dialogue=self._is_primarily_dialogue(chunk_text),
                word_count=len(chunk_words),
            )
            
            # Extract entities for child
            entities = []
            if self.nlp:
                doc = self.nlp(chunk_text)
                entities = self._extract_entities_spacy(doc)
                child_metadata.entities_mentioned = [e.name for e in entities]
            
            yield ProcessedChunk(
                chunk_id=child_id,
                text=chunk_text,
                metadata=child_metadata,
                entities=entities,
                events=[],  # Events extracted at parent level
                parent_id=parent_id,
            )
    
    def process_document(
        self,
        text: str,
        book_name: str,
        extract_events: bool = True,
    ) -> tuple[list[ProcessedChunk], nx.DiGraph]:
        """
        Process a document into hierarchical chunks with metadata.
        
        Args:
            text: Full document text
            book_name: Name of the book (for metadata)
            extract_events: Whether to extract events for GraphRAG
            
        Returns:
            Tuple of (list of ProcessedChunks, knowledge graph)
        """
        logger.info(f"Processing document: {book_name}")
        
        # Detect chapters
        chapters = self._detect_chapters(text)
        logger.info(f"Detected {len(chapters)} chapters")
        
        # Calculate total word count for narrative position
        total_words = len(text.split())
        
        # Create parent chunks
        parent_chunks: list[ProcessedChunk] = []
        child_chunks: list[ProcessedChunk] = []
        
        # Process by chapter
        for chap_idx, (chapter_num, chapter_title, start_pos) in enumerate(chapters):
            # Find chapter end
            if chap_idx + 1 < len(chapters):
                end_pos = chapters[chap_idx + 1][2]
            else:
                end_pos = len(text)
            
            chapter_text = text[start_pos:end_pos]
            chapter_words = chapter_text.split()
            chapter_word_count = len(chapter_words)
            
            # Calculate chapter's global position
            words_before = len(text[:start_pos].split())
            
            # Create parent chunks for this chapter
            chunk_size = self.config.chunking.parent_chunk_size
            overlap = self.config.chunking.parent_overlap
            step = chunk_size - overlap
            
            for i in range(0, chapter_word_count, step):
                chunk_words = chapter_words[i:i + chunk_size]
                if len(chunk_words) < chunk_size // 4:
                    continue
                    
                chunk_text = " ".join(chunk_words)
                
                # Calculate positions
                global_word_offset = words_before + i
                narrative_pos = global_word_offset / total_words if total_words > 0 else 0
                chapter_pos = i / chapter_word_count if chapter_word_count > 0 else 0
                
                # Generate parent ID
                parent_id = self._generate_chunk_id(chunk_text, narrative_pos)
                
                # Process with spaCy if available
                entities = []
                events = []
                scene_location = None
                
                if self.nlp:
                    doc = self.nlp(chunk_text)
                    entities = self._extract_entities_spacy(doc)
                    scene_location = self._extract_location(doc)
                    
                    # Extract events if requested
                    if extract_events and self.config.graph.extract_events:
                        events = self._extract_events_rule_based(
                            doc, entities, parent_id, narrative_pos, chapter_num
                        )
                        
                        # Add events to graph
                        for event in events:
                            self.entity_graph.add_edge(
                                event.subject,
                                event.object,
                                action=event.action,
                                narrative_position=event.narrative_position,
                                chunk_id=event.source_chunk_id,
                            )
                
                # Create metadata
                metadata = ChunkMetadata(
                    chunk_id=parent_id,
                    parent_id=None,  # This is a parent chunk
                    chapter_index=chapter_num,
                    chapter_title=chapter_title,
                    narrative_position=narrative_pos,
                    chapter_position=chapter_pos,
                    scene_location=scene_location,
                    time_context=self._detect_time_context(chunk_text),
                    entities_mentioned=[e.name for e in entities],
                    is_dialogue=self._is_primarily_dialogue(chunk_text),
                    word_count=len(chunk_words),
                )
                
                parent_chunk = ProcessedChunk(
                    chunk_id=parent_id,
                    text=chunk_text,
                    metadata=metadata,
                    entities=entities,
                    events=events,
                    parent_id=None,
                    child_ids=[],
                )
                
                # Create child chunks
                children = list(self._create_child_chunks(chunk_text, parent_id, metadata))
                parent_chunk.child_ids = [c.chunk_id for c in children]
                
                parent_chunks.append(parent_chunk)
                child_chunks.extend(children)
                
                # Add entities to graph
                for entity in entities:
                    if not self.entity_graph.has_node(entity.name):
                        self.entity_graph.add_node(
                            entity.name,
                            entity_type=entity.entity_type,
                            first_appearance=narrative_pos,
                        )
        
        all_chunks = parent_chunks + child_chunks
        logger.info(f"Created {len(parent_chunks)} parent chunks, {len(child_chunks)} child chunks")
        logger.info(f"Knowledge graph: {self.entity_graph.number_of_nodes()} nodes, "
                   f"{self.entity_graph.number_of_edges()} edges")
        
        return all_chunks, self.entity_graph
    
    def to_dataframe(self, chunks: list[ProcessedChunk]):
        """Convert chunks to pandas DataFrame for storage."""
        import pandas as pd
        
        records = []
        for chunk in chunks:
            records.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "parent_id": chunk.parent_id,
                "chapter_index": chunk.metadata.chapter_index,
                "chapter_title": chunk.metadata.chapter_title,
                "narrative_position": chunk.metadata.narrative_position,
                "chapter_position": chunk.metadata.chapter_position,
                "scene_location": chunk.metadata.scene_location,
                "time_context": chunk.metadata.time_context,
                "entities": chunk.metadata.entities_mentioned,
                "is_dialogue": chunk.metadata.is_dialogue,
                "word_count": chunk.metadata.word_count,
                "is_parent": chunk.parent_id is None,
            })
        
        return pd.DataFrame(records)


# Convenience function for backwards compatibility
def create_chunks_with_metadata(
    text: str,
    book_name: str,
    config: Optional[AdvancedRAGConfig] = None,
) -> tuple[list[ProcessedChunk], nx.DiGraph]:
    """
    Convenience function to chunk a document with full metadata.
    
    Args:
        text: Document text
        book_name: Name of the book
        config: Optional configuration
        
    Returns:
        Tuple of (chunks, knowledge_graph)
    """
    chunker = GraphAwareChunker(config or DEFAULT_CONFIG)
    return chunker.process_document(text, book_name)
