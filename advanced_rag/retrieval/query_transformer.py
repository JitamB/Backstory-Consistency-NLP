"""
Query Transformer with HyDE and Multi-Query Expansion

Implements pre-retrieval optimization:
1. HyDE (Hypothetical Document Embedding): Generate hypothetical answers
2. Multi-Query: Expand to semantic, keyword, temporal, and negation variants

Reduces semantic drift by matching on denser hypothetical passages.
"""

import logging
from typing import Optional

import instructor
from groq import Groq
from pydantic import BaseModel, Field

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.generation.schemas import (
    HyDEGeneration,
    MultiQueryExpansion,
    QueryIntent,
)

logger = logging.getLogger(__name__)


class HyDEPromptOutput(BaseModel):
    """Output schema for HyDE generation."""
    hypothetical_passage: str = Field(
        description="A 2-3 sentence passage that would answer the query, "
                   "written as if it's from the source book"
    )


class MultiQueryOutput(BaseModel):
    """Output schema for multi-query expansion."""
    semantic_query: str = Field(description="Rephrased query for semantic matching")
    keyword_query: str = Field(description="Keyword-focused query for BM25")
    temporal_query: Optional[str] = Field(
        default=None,
        description="Temporally-focused query if applicable"
    )
    negation_query: Optional[str] = Field(
        default=None,
        description="Query searching for contradictory evidence"
    )


class QueryTransformer:
    """
    Pre-retrieval query optimization for improved recall.
    
    Features:
    - HyDE: Generate hypothetical document, embed that instead of query
    - Multi-Query: Generate multiple query perspectives
    - Intent-aware expansion
    
    Why HyDE?
    - Query: "Harry's parents died" → Sparse, abstract
    - HyDE: "James and Lily Potter were killed by Voldemort..." → Dense
    - Dense passages match better against dense passages
    """
    
    HYDE_SYSTEM_PROMPT = """You are a passage generator. Given a question about a story,
write a 2-3 sentence passage that would answer the question. Write as if quoting from
the original book. Be specific with names, places, and events. Do not add disclaimers."""

    HYDE_USER_TEMPLATE = """Question: {query}

Write a hypothetical passage from the book that would answer this question:"""

    MULTI_QUERY_TEMPLATE = """Given this search query about a story:
"{query}"

Generate alternative queries:
1. semantic_query: Rephrase semantically (capture meaning, different words)
2. keyword_query: Extract key entity names and terms for exact matching
3. temporal_query: If about timing/sequence, add temporal focus (or null)
4. negation_query: Query to find contradicting evidence (or null)"""

    def __init__(
        self,
        llm_client: Optional[Groq] = None,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
    ):
        self.config = config
        
        # Initialize LLM client with instructor for structured outputs
        if llm_client:
            self.client = instructor.from_groq(llm_client)
        else:
            import os
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_i_1")
            if api_key:
                self.client = instructor.from_groq(Groq(api_key=api_key))
            else:
                logger.warning("No Groq API key found. Query transformation disabled.")
                self.client = None
    
    def generate_hyde(
        self,
        query: str,
        num_hypotheticals: int = 1,
    ) -> list[str]:
        """
        Generate hypothetical document(s) for HyDE.
        
        Args:
            query: Original search query
            num_hypotheticals: Number of hypothetical passages to generate
            
        Returns:
            List of hypothetical passage strings
        """
        if not self.client:
            logger.warning("HyDE unavailable: no LLM client")
            return [query]  # Fallback to original query
        
        hypotheticals = []
        
        for i in range(num_hypotheticals):
            try:
                result = self.client.chat.completions.create(
                    model=self.config.llm.model_name,
                    messages=[
                        {"role": "system", "content": self.HYDE_SYSTEM_PROMPT},
                        {"role": "user", "content": self.HYDE_USER_TEMPLATE.format(query=query)},
                    ],
                    response_model=HyDEPromptOutput,
                    temperature=0.3 if i > 0 else 0.0,  # Vary for diversity
                    max_tokens=256,
                )
                hypotheticals.append(result.hypothetical_passage)
                logger.debug(f"HyDE {i+1}: {result.hypothetical_passage[:100]}...")
            except Exception as e:
                logger.error(f"HyDE generation failed: {e}")
                hypotheticals.append(query)  # Fallback
        
        return hypotheticals
    
    def expand_multi_query(
        self,
        query: str,
        query_intent: Optional[QueryIntent] = None,
    ) -> MultiQueryExpansion:
        """
        Expand query into multiple perspectives.
        
        Args:
            query: Original query
            query_intent: Optional pre-classified intent
            
        Returns:
            MultiQueryExpansion with all query variants
        """
        if not self.client:
            return MultiQueryExpansion(
                original_query=query,
                semantic_variant=query,
                keyword_variant=query,
            )
        
        try:
            result = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {"role": "user", "content": self.MULTI_QUERY_TEMPLATE.format(query=query)},
                ],
                response_model=MultiQueryOutput,
                temperature=0.0,
                max_tokens=512,
            )
            
            return MultiQueryExpansion(
                original_query=query,
                semantic_variant=result.semantic_query,
                keyword_variant=result.keyword_query,
                temporal_variant=result.temporal_query,
                negation_variant=result.negation_query,
            )
        except Exception as e:
            logger.error(f"Multi-query expansion failed: {e}")
            return MultiQueryExpansion(
                original_query=query,
                semantic_variant=query,
                keyword_variant=query,
            )
    
    def expand(
        self,
        query: str,
        use_hyde: bool = True,
        use_multi_query: bool = True,
        num_hypotheticals: int = 2,
    ) -> dict:
        """
        Full query expansion combining HyDE and Multi-Query.
        
        Args:
            query: Original query
            use_hyde: Whether to use HyDE
            use_multi_query: Whether to use multi-query expansion
            num_hypotheticals: Number of HyDE passages
            
        Returns:
            Dict with 'queries' list and 'expansion_info'
        """
        queries = [query]  # Always include original
        expansion_info = {"original": query}
        
        # HyDE expansion
        if use_hyde:
            hyde_passages = self.generate_hyde(query, num_hypotheticals)
            queries.extend(hyde_passages)
            expansion_info["hyde"] = hyde_passages
        
        # Multi-query expansion
        if use_multi_query:
            multi = self.expand_multi_query(query)
            
            if multi.semantic_variant and multi.semantic_variant != query:
                queries.append(multi.semantic_variant)
            if multi.keyword_variant and multi.keyword_variant != query:
                queries.append(multi.keyword_variant)
            if multi.temporal_variant:
                queries.append(multi.temporal_variant)
            if multi.negation_variant:
                queries.append(multi.negation_variant)
            
            expansion_info["multi_query"] = {
                "semantic": multi.semantic_variant,
                "keyword": multi.keyword_variant,
                "temporal": multi.temporal_variant,
                "negation": multi.negation_variant,
            }
        
        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q and q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        logger.info(f"Expanded {query[:50]}... into {len(unique_queries)} queries")
        
        return {
            "queries": unique_queries,
            "expansion_info": expansion_info,
        }


def create_query_transformer(
    api_key: Optional[str] = None,
    config: Optional[AdvancedRAGConfig] = None,
) -> QueryTransformer:
    """Factory function to create QueryTransformer."""
    client = Groq(api_key=api_key) if api_key else None
    return QueryTransformer(client, config or DEFAULT_CONFIG)
