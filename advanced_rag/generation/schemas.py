"""
Pydantic schemas for structured LLM outputs.
Replaces fragile JSON parsing with guaranteed schema compliance.

Uses `instructor` library for automatic retries on schema violations.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum


# ============================================
# Verdicts & Classification Enums
# ============================================

class Verdict(str, Enum):
    """Possible outcomes for claim verification."""
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


class QueryIntent(str, Enum):
    """
    Query intent for temporal bias in retrieval.
    - CURRENT_STATE: Boost late-narrative chunks (e.g., "Who won in the end?")
    - ORIGIN_STORY: Boost early-narrative chunks (e.g., "Where did Harry grow up?")
    - GENERAL: No temporal bias
    """
    CURRENT_STATE = "current_state"
    ORIGIN_STORY = "origin_story"
    GENERAL = "general"


class ClaimType(str, Enum):
    """Types of claims for specialized verification."""
    TEMPORAL = "TEMPORAL"
    RELATIONSHIP = "RELATIONSHIP"
    LOCATION = "LOCATION"
    TRAIT = "TRAIT"
    EVENT = "EVENT"
    GENERAL = "GENERAL"


# ============================================
# Entity & Event Schemas (for GraphRAG)
# ============================================

class EntityNode(BaseModel):
    """An entity node in the knowledge graph."""
    name: str = Field(description="Canonical entity name (e.g., 'Harry Potter')")
    entity_type: Literal["CHARACTER", "LOCATION", "OBJECT", "ORGANIZATION"] = Field(
        description="Type of entity"
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names/references (e.g., ['The Boy Who Lived', 'Harry'])"
    )
    first_appearance: float = Field(
        ge=0.0, le=1.0,
        default=0.0,
        description="Narrative position of first mention (0.0-1.0)"
    )


class EventEdge(BaseModel):
    """
    An event edge connecting entities in the knowledge graph.
    Event-centric design: captures who-did-what-to-whom with temporal context.
    """
    subject: str = Field(description="Entity performing the action")
    action: str = Field(description="The action/event verb (e.g., 'defeated', 'married', 'betrayed')")
    object: str = Field(description="Entity receiving the action")
    narrative_position: float = Field(
        ge=0.0, le=1.0,
        description="When this event occurs in the narrative (0.0=start, 1.0=end)"
    )
    chapter_index: int = Field(
        default=0,
        description="Chapter where this event occurs"
    )
    source_chunk_id: str = Field(
        description="ID of the chunk where this event was extracted"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        default=0.8,
        description="Confidence in event extraction accuracy"
    )


class EntityExtractionResult(BaseModel):
    """Result of entity and event extraction from a chunk."""
    entities: list[EntityNode] = Field(default_factory=list)
    events: list[EventEdge] = Field(default_factory=list)
    key_relationships: list[str] = Field(
        default_factory=list,
        description="Summary of key relationships in this chunk"
    )


# ============================================
# Chunk Metadata Schemas
# ============================================

class ChunkMetadata(BaseModel):
    """
    Rich metadata for each chunk enabling temporal-aware retrieval.
    Solves the 'narrative time' problem where facts change over the story.
    """
    chunk_id: str = Field(description="Unique identifier for this chunk")
    parent_id: Optional[str] = Field(
        default=None,
        description="Parent chunk ID for hierarchical retrieval"
    )
    book_name: Optional[str] = Field(
        default=None,
        description="Name of the source book"
    )
    chapter_index: int = Field(
        ge=0,
        description="Chapter number (0-indexed)"
    )
    chapter_title: Optional[str] = Field(
        default=None,
        description="Chapter title if detected"
    )
    narrative_position: float = Field(
        ge=0.0, le=1.0,
        description="Global position in book (0.0=start, 1.0=end)"
    )
    chapter_position: float = Field(
        ge=0.0, le=1.0,
        default=0.0,
        description="Position within current chapter"
    )
    scene_location: Optional[str] = Field(
        default=None,
        description="Detected location of this scene"
    )
    time_context: Optional[str] = Field(
        default=None,
        description="Temporal context (e.g., 'night', 'morning', 'flashback')"
    )
    entities_mentioned: list[str] = Field(
        default_factory=list,
        description="Entity names mentioned in this chunk"
    )
    is_dialogue: bool = Field(
        default=False,
        description="Whether this chunk is primarily dialogue"
    )
    word_count: int = Field(default=0, description="Number of words in chunk")


# ============================================
# Verification Schemas (Chain-of-Verification)
# ============================================

class SubClaimVerification(BaseModel):
    """Verification result for a single atomic sub-claim."""
    claim: str = Field(description="The atomic sub-claim being verified")
    evidence: Optional[str] = Field(
        default=None,
        description="Exact quoted text from passage, or None if not found"
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Passage IDs cited (e.g., ['passage_1', 'passage_3'])"
    )
    verdict: Verdict = Field(description="Verification verdict for this sub-claim")
    reasoning: str = Field(
        default="",
        description="Brief explanation of the verdict"
    )


class VerificationResult(BaseModel):
    """
    Complete structured output for Chain-of-Verification.
    Using instructor library guarantees schema compliance with auto-retries.
    """
    sub_claims: list[SubClaimVerification] = Field(
        description="Individual verification for each atomic sub-claim"
    )
    verdict: Verdict = Field(description="Overall verification verdict")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Overall confidence in verdict"
    )
    explicit_contradictions: list[str] = Field(
        default_factory=list,
        description="Exact quoted contradictory text from evidence"
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="What evidence would be needed but is missing"
    )
    temporal_consistency: bool = Field(
        default=True,
        description="Whether evidence is temporally consistent with claim"
    )


class IntentClassification(BaseModel):
    """
    Query intent classification for temporal bias routing.
    Used by query router to determine appropriate retrieval strategy.
    """
    intent: QueryIntent = Field(description="Classified query intent")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in classification"
    )
    reasoning: str = Field(description="Brief explanation for classification")
    temporal_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that influenced classification"
    )


# ============================================
# Claim Decomposition Schemas
# ============================================

class DecomposedClaim(BaseModel):
    """A single atomic claim extracted from a backstory."""
    text: str = Field(description="The atomic claim statement")
    claim_type: ClaimType = Field(description="Category of the claim")
    queries: list[str] = Field(
        description="Search queries for retrieval (keyword, semantic, anti-evidence)"
    )
    expected_temporal_range: Optional[tuple[float, float]] = Field(
        default=None,
        description="Expected narrative position range (start, end) for evidence"
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="Key entities mentioned in this claim"
    )


class BackstoryDecomposition(BaseModel):
    """Complete decomposition of a backstory into verifiable claims."""
    character_name: str = Field(description="Name of the character")
    book_title: str = Field(description="Title of the source book")
    claims: list[DecomposedClaim] = Field(description="List of atomic claims")
    entity_mentions: list[str] = Field(
        default_factory=list,
        description="All entities mentioned in the backstory"
    )


# ============================================
# Query Transformation Schemas
# ============================================

class HyDEGeneration(BaseModel):
    """Hypothetical document for HyDE query expansion."""
    hypothetical_passage: str = Field(
        description="A hypothetical passage that would answer the query"
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="Key terms from the hypothetical passage"
    )


class MultiQueryExpansion(BaseModel):
    """Multiple query variants for robust retrieval."""
    original_query: str = Field(description="Original query")
    semantic_variant: str = Field(description="Rephrased for semantic matching")
    keyword_variant: str = Field(description="Keyword-focused variant for BM25")
    temporal_variant: Optional[str] = Field(
        default=None,
        description="Temporally-focused variant if applicable"
    )
    negation_variant: Optional[str] = Field(
        default=None,
        description="Variant searching for contradictory evidence"
    )
