"""
Advanced RAG Pipeline - Unified Entry Point

Orchestrates the complete production-grade RAG system:
1. Document ingestion with hierarchical chunking
2. Hybrid index building (semantic + BM25)
3. Confidence-based query routing
4. HyDE and multi-query expansion
5. Hybrid retrieval with RRF fusion
6. Graph-based entity retrieval
7. Cross-encoder reranking
8. Chain-of-Verification generation
9. Self-consistency decoding

Target: <1% hallucination rate with ~4s average latency.
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()  # Load .env file

import pandas as pd
from groq import Groq

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.indexing.graph_chunker import GraphAwareChunker, ProcessedChunk
from advanced_rag.retrieval.hybrid_retriever import HybridRetriever
from advanced_rag.retrieval.query_router import QueryRouter, AdaptivePipeline, PipelineMode
from advanced_rag.retrieval.query_transformer import QueryTransformer
from advanced_rag.retrieval.graph_retriever import GraphRetriever
from advanced_rag.retrieval.agentic_retriever import AgenticRetriever
from advanced_rag.reranking.cross_encoder_reranker import AdvancedReranker
from advanced_rag.generation.grounded_generator import GroundedGenerator
from advanced_rag.generation.schemas import (
    Verdict,
    VerificationResult,
    BackstoryDecomposition,
    QueryIntent,
)

logger = logging.getLogger(__name__)


@dataclass 
class VerificationOutput:
    """Complete verification output for a backstory."""
    character_name: str
    book_title: str
    is_consistent: bool
    confidence: float
    verdict: Verdict
    verification_result: VerificationResult
    claims_verified: int
    claims_contradicted: int
    claims_insufficient: int
    rationale: str
    latency_ms: int


class AdvancedRAGPipeline:
    """
    Production-grade RAG pipeline for backstory consistency verification.
    
    Usage:
        pipeline = AdvancedRAGPipeline()
        pipeline.ingest_books("./Dataset/Books")
        result = pipeline.verify_backstory(backstory, character, book)
    """
    
    def __init__(
        self,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
        api_key: Optional[str] = None,
    ):
        self.config = config
        
        # Initialize Groq client
        api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_i_1")
        self.llm_client = Groq(api_key=api_key) if api_key else None
        
        # Initialize components
        self.chunker = GraphAwareChunker(config)
        self.retriever = HybridRetriever(config)
        self.graph_retriever = GraphRetriever(config=config)
        self.reranker = AdvancedReranker(config)
        self.query_transformer = QueryTransformer(self.llm_client, config)
        self.generator = GroundedGenerator(self.llm_client, config)
        
        # Router and adaptive pipeline
        self.router = QueryRouter(config)
        
        # State
        self.is_indexed = False
        self.chunks: list[ProcessedChunk] = []
        
        logger.info("AdvancedRAGPipeline initialized")
    
    def ingest_books(
        self,
        data_dir: str,
        extract_events: bool = True,
    ) -> pd.DataFrame:
        """
        Ingest books from directory and build indices.
        
        Args:
            data_dir: Directory containing .txt book files
            extract_events: Whether to extract events for GraphRAG
            
        Returns:
            DataFrame with chunk information
        """
        logger.info(f"Ingesting books from {data_dir}")
        
        all_chunks = []
        
        # Read and process each book
        for filename in os.listdir(data_dir):
            if not filename.endswith('.txt'):
                continue
            
            filepath = os.path.join(data_dir, filename)
            book_name = os.path.splitext(filename)[0]
            
            logger.info(f"Processing: {book_name}")
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Chunk with metadata
            chunks, entity_graph = self.chunker.process_document(
                text, book_name, extract_events=extract_events
            )
            
            # Add book name to each chunk
            for chunk in chunks:
                chunk.metadata.book_name = book_name
            
            all_chunks.extend(chunks)
            
            # Update graph retriever
            self.graph_retriever.build_from_chunks(
                [{"chunk_id": c.chunk_id, 
                  "entities": c.entities, 
                  "events": c.events} for c in chunks],
                entity_graph,
            )
        
        self.chunks = all_chunks
        
        # Build hybrid index
        chunk_dicts = []
        for c in all_chunks:
            chunk_dicts.append({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
                "parent_id": c.parent_id,
                "is_parent": c.parent_id is None,
            })
        
        self.retriever.build_index(chunk_dicts)
        self.is_indexed = True
        
        logger.info(f"Ingested {len(all_chunks)} chunks from {data_dir}")
        
        return self.chunker.to_dataframe(all_chunks)
    
    def verify_backstory(
        self,
        backstory: str,
        character_name: str,
        book_title: str,
        force_mode: Optional[PipelineMode] = None,
    ) -> VerificationOutput:
        """
        Verify a character backstory against book evidence.
        
        Args:
            backstory: The backstory text to verify
            character_name: Name of the character
            book_title: Title of the source book
            force_mode: Force specific pipeline mode
            
        Returns:
            VerificationOutput with complete results
        """
        import time
        start_time = time.time()
        
        if not self.is_indexed:
            raise ValueError("Pipeline not indexed. Call ingest_books() first.")
        
        # 1. Decompose backstory into claims
        decomposition = self.generator.decompose_backstory(
            backstory, character_name, book_title
        )
        
        # 2. Verify each claim
        all_results = []
        
        for claim in decomposition.claims:
            # Determine routing
            confidence = self.router.estimate_confidence(claim.text, self.retriever)
            intent = self.router.classify_intent(claim.text)
            decision = self.router.route(claim.text, confidence, intent)
            
            if force_mode:
                decision.mode = force_mode
            
            # Expand queries based on mode
            if decision.skip_hyde:
                queries = claim.queries
            else:
                expansion = self.query_transformer.expand(claim.text)
                queries = expansion.get("queries", claim.queries)
            
            # Retrieve evidence
            if len(queries) > 1:
                results = self.retriever.multi_query_search(
                    queries,
                    book_filter=book_title,
                    query_intent=intent,
                )
            else:
                results = self.retriever.search(
                    claim.text,
                    book_filter=book_title,
                    query_intent=intent,
                )
            
            # Augment with graph results
            graph_result = self.graph_retriever.search(
                claim.text,
                narrative_range=claim.expected_temporal_range,
            )
            
            # Combine chunk IDs
            graph_chunk_ids = set(graph_result.chunk_ids)
            for r in results:
                if r.chunk_id in graph_chunk_ids:
                    r.final_score += 0.1  # Boost graph matches
            
            # Rerank
            results = self.reranker.rerank(claim.text, results, query_intent=intent)
            
            # Format evidence for verification
            evidence = [
                {
                    "chunk_id": r.chunk_id,
                    "text": r.parent_text or r.text,
                    "narrative_position": r.metadata.narrative_position,
                }
                for r in results[:5]
            ]
            
            # Verify claim
            verification = self.generator.verify_claim(
                claim.text,
                evidence,
                use_self_consistency=True,
            )
            
            all_results.append(verification)
        
        # 3. Aggregate results
        final_verdict = Verdict.SUPPORT
        contradicted = 0
        insufficient = 0
        rationales = []
        
        for result in all_results:
            if result.verdict == Verdict.CONTRADICT:
                final_verdict = Verdict.CONTRADICT
                contradicted += 1
            elif result.verdict == Verdict.INSUFFICIENT:
                insufficient += 1
            
            for sub in result.sub_claims:
                if sub.reasoning:
                    rationales.append(sub.reasoning)
            
            if result.explicit_contradictions:
                rationales.extend(result.explicit_contradictions)
        
        # If majority insufficient, overall is insufficient
        if insufficient > len(all_results) / 2 and final_verdict != Verdict.CONTRADICT:
            final_verdict = Verdict.INSUFFICIENT
        
        # Calculate confidence
        avg_confidence = sum(r.confidence for r in all_results) / len(all_results) if all_results else 0
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return VerificationOutput(
            character_name=character_name,
            book_title=book_title,
            is_consistent=final_verdict == Verdict.SUPPORT,
            confidence=avg_confidence,
            verdict=final_verdict,
            verification_result=all_results[0] if all_results else None,
            claims_verified=len(all_results) - contradicted - insufficient,
            claims_contradicted=contradicted,
            claims_insufficient=insufficient,
            rationale=" | ".join(rationales[:5]),
            latency_ms=elapsed_ms,
        )
    
    def run_validation(
        self,
        validation_csv: str,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run validation on a CSV dataset.
        
        Args:
            validation_csv: Path to CSV with columns: content, char, book_name, label
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame with predictions
        """
        df = pd.read_csv(validation_csv)
        if limit:
            df = df.head(limit)
        
        results = []
        
        from tqdm import tqdm
        for i, row in tqdm(df.iterrows(), total=len(df)):
            backstory = row.get('content') or row.get('backstory')
            character = row.get('char') or row.get('Character') or "Unknown"
            book = row.get('book_name') or row.get('Book')
            truth = row.get('label', 1)
            
            try:
                output = self.verify_backstory(backstory, character, book)
                pred = 1 if output.is_consistent else 0
            except Exception as e:
                logger.error(f"Error on row {i}: {e}")
                pred = 1  # Default to consistent on error
                output = None
            
            results.append({
                'id': row.get('id', i),
                'truth': truth,
                'prediction': pred,
                'confidence': output.confidence if output else 0,
                'verdict': output.verdict.value if output else 'ERROR',
                'latency_ms': output.latency_ms if output else 0,
            })
        
        results_df = pd.DataFrame(results)
        
        # Print metrics
        from sklearn.metrics import accuracy_score, classification_report
        print(f"\nAccuracy: {accuracy_score(results_df['truth'], results_df['prediction']):.2%}")
        print(classification_report(
            results_df['truth'], 
            results_df['prediction'],
            target_names=["Contradiction", "Consistent"]
        ))
        
        return results_df


def create_pipeline(
    config: Optional[AdvancedRAGConfig] = None,
    api_key: Optional[str] = None,
) -> AdvancedRAGPipeline:
    """Factory function to create pipeline."""
    return AdvancedRAGPipeline(config or DEFAULT_CONFIG, api_key)
