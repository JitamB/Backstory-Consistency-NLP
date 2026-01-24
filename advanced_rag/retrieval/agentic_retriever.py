"""
Agentic Retriever with Self-Improving Retrieval Loop

Implements LLM-driven retrieval that decides when to fetch more data:
1. Initial retrieval
2. LLM evaluates: "Do I have enough evidence?"
3. If NO: Generate refined query based on gaps
4. Re-retrieve with new query
5. Repeat until confident OR max_iterations

Catches cases where initial query missed relevant passages.
"""

import logging
from typing import Optional
from dataclasses import dataclass

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.retrieval.hybrid_retriever import HybridRetriever, RetrievalResult
from advanced_rag.retrieval.query_transformer import QueryTransformer
from advanced_rag.generation.schemas import QueryIntent

logger = logging.getLogger(__name__)


@dataclass
class AgenticIteration:
    """Record of one iteration in the agentic loop."""
    iteration: int
    query: str
    results_count: int
    max_score: float
    gaps_identified: list[str]
    refined_queries: list[str]


@dataclass
class AgenticRetrievalResult:
    """Final result from agentic retrieval."""
    results: list[RetrievalResult]
    iterations: list[AgenticIteration]
    final_confidence: float
    total_iterations: int
    stopped_reason: str  # "confident", "max_iterations", "no_improvement"


class AgenticRetriever:
    """
    Self-improving retrieval with LLM-guided query refinement.
    
    The agent loop:
    1. Initial search with user query
    2. Evaluate result quality (score, count, diversity)
    3. If insufficient, identify gaps and generate refined queries
    4. Re-search with refined queries
    5. Combine and deduplicate results
    6. Repeat until confident or max iterations reached
    
    This addresses SEMANTIC DRIFT by allowing the LLM to guide
    its own information needs when initial queries miss.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        query_transformer: Optional[QueryTransformer] = None,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
    ):
        self.retriever = retriever
        self.query_transformer = query_transformer
        self.config = config
        
        # Thresholds
        self.confidence_threshold = 0.7
        self.min_results = 3
        self.max_iterations = config.llm.max_agent_iterations
    
    def _evaluate_results(
        self,
        results: list[RetrievalResult],
        required_entities: list[str] = None,
    ) -> tuple[float, list[str]]:
        """
        Evaluate retrieval quality and identify gaps.
        
        Returns:
            Tuple of (confidence_score, list_of_gaps)
        """
        gaps = []
        
        # Count check
        if len(results) < self.min_results:
            gaps.append(f"Only {len(results)} results found, need at least {self.min_results}")
        
        # Score check
        if results:
            max_score = results[0].final_score
            if max_score < 0.5:
                gaps.append(f"Low relevance scores (max: {max_score:.2f})")
            
            # Score diversity check
            scores = [r.final_score for r in results[:5]]
            if len(scores) > 1 and (max(scores) - min(scores)) < 0.1:
                gaps.append("Results lack score diversity - may be missing relevant content")
        else:
            gaps.append("No results found")
            return 0.0, gaps
        
        # Entity coverage check
        if required_entities:
            found_entities = set()
            for r in results:
                for e in r.metadata.entities_mentioned:
                    found_entities.add(e.lower())
            
            missing = [e for e in required_entities 
                      if e.lower() not in found_entities]
            if missing:
                gaps.append(f"Missing entity mentions: {missing}")
        
        # Temporal coverage check
        positions = [r.metadata.narrative_position for r in results]
        if positions:
            pos_range = max(positions) - min(positions)
            if pos_range < 0.3:
                gaps.append(f"Results clustered in narrow narrative range ({pos_range:.2f})")
        
        # Calculate confidence
        confidence = results[0].final_score if results else 0.0
        confidence *= (1 - len(gaps) * 0.1)  # Penalize for gaps
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence, gaps
    
    def _generate_refined_queries(
        self,
        original_query: str,
        gaps: list[str],
        previous_results: list[RetrievalResult],
    ) -> list[str]:
        """
        Generate refined queries based on identified gaps.
        
        Uses heuristics and optionally LLM for query refinement.
        """
        refined = []
        
        for gap in gaps:
            if "Missing entity" in gap:
                # Extract missing entities and search for them directly
                import re
                entities = re.findall(r'\[([^\]]+)\]', gap)
                for entity in entities[:2]:
                    refined.append(f"{entity} {original_query}")
            
            elif "narrow narrative range" in gap:
                # Add temporal qualifiers
                refined.append(f"beginning early {original_query}")
                refined.append(f"ending conclusion {original_query}")
            
            elif "Low relevance" in gap:
                # Try simpler, more focused query
                words = original_query.split()[:5]
                refined.append(" ".join(words))
        
        # Use query transformer if available
        if self.query_transformer and not refined:
            expansion = self.query_transformer.expand(original_query)
            refined.extend(expansion.get("queries", [])[:2])
        
        return refined[:3]  # Limit refined queries
    
    def search(
        self,
        query: str,
        book_filter: Optional[str] = None,
        query_intent: QueryIntent = QueryIntent.GENERAL,
        required_entities: list[str] = None,
        max_iterations: int = None,
    ) -> AgenticRetrievalResult:
        """
        Perform agentic retrieval with self-improvement loop.
        
        Args:
            query: Initial search query
            book_filter: Optional book filter
            query_intent: Query intent for temporal bias
            required_entities: Entities that should appear in results
            max_iterations: Override for max iterations
            
        Returns:
            AgenticRetrievalResult with all results and iteration history
        """
        max_iter = max_iterations or self.max_iterations
        iterations: list[AgenticIteration] = []
        all_results: dict[str, RetrievalResult] = {}
        current_query = query
        
        for i in range(max_iter):
            logger.info(f"Agentic iteration {i+1}/{max_iter}: {current_query[:50]}...")
            
            # Search
            results = self.retriever.search(
                current_query,
                top_k=self.config.retrieval.initial_candidates,
                book_filter=book_filter,
                query_intent=query_intent,
            )
            
            # Merge results (keep highest score for duplicates)
            for r in results:
                if r.chunk_id not in all_results or r.final_score > all_results[r.chunk_id].final_score:
                    all_results[r.chunk_id] = r
            
            # Evaluate
            confidence, gaps = self._evaluate_results(
                list(all_results.values()),
                required_entities,
            )
            
            # Record iteration
            iteration = AgenticIteration(
                iteration=i + 1,
                query=current_query,
                results_count=len(results),
                max_score=results[0].final_score if results else 0.0,
                gaps_identified=gaps,
                refined_queries=[],
            )
            
            # Check stopping conditions
            if confidence >= self.confidence_threshold:
                iteration.refined_queries = []
                iterations.append(iteration)
                logger.info(f"Stopping: Confidence {confidence:.2f} >= threshold")
                return AgenticRetrievalResult(
                    results=sorted(all_results.values(), 
                                 key=lambda x: x.final_score, reverse=True),
                    iterations=iterations,
                    final_confidence=confidence,
                    total_iterations=i + 1,
                    stopped_reason="confident",
                )
            
            if not gaps:
                iteration.refined_queries = []
                iterations.append(iteration)
                logger.info("Stopping: No gaps identified")
                return AgenticRetrievalResult(
                    results=sorted(all_results.values(),
                                 key=lambda x: x.final_score, reverse=True),
                    iterations=iterations,
                    final_confidence=confidence,
                    total_iterations=i + 1,
                    stopped_reason="no_improvement",
                )
            
            # Generate refined queries for next iteration
            refined = self._generate_refined_queries(query, gaps, results)
            iteration.refined_queries = refined
            iterations.append(iteration)
            
            if refined:
                current_query = refined[0]  # Use first refined query
            else:
                logger.info("Stopping: No refined queries generated")
                break
        
        # Max iterations reached
        final_results = sorted(all_results.values(),
                              key=lambda x: x.final_score, reverse=True)
        final_confidence, _ = self._evaluate_results(final_results, required_entities)
        
        return AgenticRetrievalResult(
            results=final_results,
            iterations=iterations,
            final_confidence=final_confidence,
            total_iterations=max_iter,
            stopped_reason="max_iterations",
        )
