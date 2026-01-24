"""
Grounded Generator with Chain-of-Verification

Implements citation-grounded generation with:
1. Structured Pydantic outputs via instructor
2. Self-consistency decoding (3x verification)
3. Citation validation
4. Confidence-based flagging

Target: <1% hallucination rate through mandatory evidence grounding.
"""

import logging
from typing import Optional, List
import time

import instructor
from groq import Groq

from advanced_rag.config import AdvancedRAGConfig, DEFAULT_CONFIG
from advanced_rag.generation.schemas import (
    Verdict,
    VerificationResult,
    SubClaimVerification,
    BackstoryDecomposition,
    DecomposedClaim,
    ClaimType,
)
from advanced_rag.generation.cove_prompts import (
    COVE_VERIFICATION_TEMPLATE,
    CLAIM_DECOMPOSITION_TEMPLATE,
    SELF_CONSISTENCY_TEMPLATE,
)

logger = logging.getLogger(__name__)


class GroundedGenerator:
    """
    Citation-grounded verification generator.
    
    Features:
    - Pydantic structured outputs (no fragile JSON parsing)
    - Chain-of-Verification prompting
    - Self-consistency decoding (majority voting)
    - Citation validation post-processing
    - Agentic re-retrieval trigger on low confidence
    """
    
    def __init__(
        self,
        llm_client: Optional[Groq] = None,
        config: AdvancedRAGConfig = DEFAULT_CONFIG,
    ):
        self.config = config
        
        # Initialize instructor-wrapped client
        if llm_client:
            self.client = instructor.from_groq(llm_client)
            self.raw_client = llm_client
        else:
            import os
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_i_1")
            if api_key:
                self.raw_client = Groq(api_key=api_key)
                self.client = instructor.from_groq(self.raw_client)
            else:
                logger.warning("No Groq API key found. Generation disabled.")
                self.client = None
                self.raw_client = None
    
    def _format_evidence(self, evidence: list[dict]) -> str:
        """Format evidence chunks with IDs for the prompt."""
        formatted = []
        for i, e in enumerate(evidence):
            chunk_id = e.get("chunk_id", f"passage_{i+1}")
            text = e.get("text", "")
            position = e.get("narrative_position", 0)
            
            formatted.append(
                f"[{chunk_id}] (Position: {position:.2f})\n{text}\n"
            )
        return "\n".join(formatted)
    
    def _validate_citations(
        self,
        result: VerificationResult,
        evidence: list[dict],
    ) -> VerificationResult:
        """
        Validate that all citations reference actual evidence.
        
        Removes claims with invalid citations and adjusts confidence.
        """
        valid_ids = {e.get("chunk_id", f"passage_{i}") for i, e in enumerate(evidence)}
        
        validated_subclaims = []
        invalid_citations = 0
        
        for subclaim in result.sub_claims:
            valid_cites = [c for c in subclaim.citations if c in valid_ids]
            if subclaim.citations and not valid_cites:
                # All citations invalid - mark as unfounded
                invalid_citations += 1
                subclaim.evidence = None
                subclaim.citations = []
                subclaim.verdict = Verdict.INSUFFICIENT
                subclaim.reasoning += " [WARNING: Invalid citations removed]"
            else:
                subclaim.citations = valid_cites
            validated_subclaims.append(subclaim)
        
        # Adjust confidence if citations were invalid
        if invalid_citations > 0:
            penalty = invalid_citations * 0.1
            result.confidence = max(0.0, result.confidence - penalty)
            logger.warning(f"Removed {invalid_citations} invalid citations, "
                         f"confidence adjusted to {result.confidence:.2f}")
        
        result.sub_claims = validated_subclaims
        return result
    
    def verify_claim(
        self,
        claim: str,
        evidence: list[dict],
        use_self_consistency: bool = True,
    ) -> VerificationResult:
        """
        Verify a claim against evidence using CoVe.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence chunk dicts
            use_self_consistency: Whether to use self-consistency decoding
            
        Returns:
            VerificationResult with structured verdict
        """
        if not self.client:
            logger.error("No LLM client available")
            return VerificationResult(
                sub_claims=[],
                verdict=Verdict.INSUFFICIENT,
                confidence=0.0,
                missing_information=["LLM client not configured"],
            )
        
        evidence_text = self._format_evidence(evidence)
        prompt = COVE_VERIFICATION_TEMPLATE.format(
            claim=claim,
            evidence_with_ids=evidence_text,
        )
        
        if use_self_consistency:
            return self._verify_with_consistency(claim, evidence, prompt)
        else:
            return self._single_verification(prompt, evidence)
    
    def _single_verification(
        self,
        prompt: str,
        evidence: list[dict],
    ) -> VerificationResult:
        """Single verification pass."""
        try:
            result = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {"role": "system", "content": "You are a fact-checker. Output structured verification."},
                    {"role": "user", "content": prompt},
                ],
                response_model=VerificationResult,
                temperature=0.0,
                max_tokens=self.config.llm.max_tokens,
            )
            
            # Validate citations
            result = self._validate_citations(result, evidence)
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                sub_claims=[],
                verdict=Verdict.INSUFFICIENT,
                confidence=0.0,
                missing_information=[f"Verification error: {str(e)}"],
            )
    
    def _verify_with_consistency(
        self,
        claim: str,
        evidence: list[dict],
        prompt: str,
    ) -> VerificationResult:
        """
        Self-consistency decoding: Run verification multiple times and vote.
        
        If verdicts disagree, flag for human review or lower confidence.
        """
        n_samples = self.config.llm.self_consistency_samples
        results: List[VerificationResult] = []
        
        for i in range(n_samples):
            temperature = 0.0 if i == 0 else self.config.llm.consistency_temperature
            
            try:
                result = self.client.chat.completions.create(
                    model=self.config.llm.model_name,
                    messages=[
                        {"role": "system", "content": "You are a fact-checker. Output structured verification."},
                        {"role": "user", "content": prompt},
                    ],
                    response_model=VerificationResult,
                    temperature=temperature,
                    max_tokens=self.config.llm.max_tokens,
                )
                results.append(result)
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Consistency sample {i+1} failed: {e}")
        
        if not results:
            return VerificationResult(
                sub_claims=[],
                verdict=Verdict.INSUFFICIENT,
                confidence=0.0,
                missing_information=["All verification attempts failed"],
            )
        
        # Majority voting on verdict
        verdict_counts = {}
        for r in results:
            v = r.verdict
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        
        majority_verdict = max(verdict_counts, key=verdict_counts.get)
        agreement_ratio = verdict_counts[majority_verdict] / len(results)
        
        # Use most detailed result as base
        base_result = max(results, key=lambda r: len(r.sub_claims))
        base_result.verdict = majority_verdict
        
        # Adjust confidence based on agreement
        if agreement_ratio < 1.0:
            base_result.confidence *= agreement_ratio
            base_result.missing_information.append(
                f"Self-consistency: {agreement_ratio:.0%} agreement on verdict"
            )
        
        # Validate citations
        base_result = self._validate_citations(base_result, evidence)
        
        logger.info(f"Self-consistency: {majority_verdict} "
                   f"({agreement_ratio:.0%} agreement, conf={base_result.confidence:.2f})")
        
        return base_result
    
    def decompose_backstory(
        self,
        backstory: str,
        character_name: str,
        book_title: str,
    ) -> BackstoryDecomposition:
        """
        Decompose a backstory into atomic, verifiable claims.
        
        Returns structured decomposition with search queries for each claim.
        """
        if not self.client:
            logger.error("No LLM client available")
            return BackstoryDecomposition(
                character_name=character_name,
                book_title=book_title,
                claims=[],
            )
        
        prompt = CLAIM_DECOMPOSITION_TEMPLATE.format(
            character_name=character_name,
            backstory=backstory,
        )
        
        try:
            result = self.client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_model=BackstoryDecomposition,
                temperature=0.1,
                max_tokens=2048,
            )
            
            # Ensure book title is set
            result.book_title = book_title
            
            logger.info(f"Decomposed backstory into {len(result.claims)} claims")
            return result
            
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            # Fallback: single claim
            return BackstoryDecomposition(
                character_name=character_name,
                book_title=book_title,
                claims=[
                    DecomposedClaim(
                        text=backstory,
                        claim_type=ClaimType.GENERAL,
                        queries=[backstory],
                    )
                ],
            )
    
    def should_re_retrieve(self, result: VerificationResult) -> bool:
        """
        Determine if re-retrieval is needed based on verification result.
        
        Triggers agentic retrieval if:
        - Confidence below threshold
        - Missing information identified
        - Too many INSUFFICIENT sub-claims
        """
        # Low overall confidence
        if result.confidence < 0.6:
            return True
        
        # Too many sub-claims lack evidence
        if result.sub_claims:
            insufficient = sum(1 for s in result.sub_claims 
                             if s.verdict == Verdict.INSUFFICIENT)
            if insufficient / len(result.sub_claims) > 0.3:
                return True
        
        # Missing information identified
        if len(result.missing_information) > 2:
            return True
        
        return False
    
    def get_re_retrieval_queries(self, result: VerificationResult) -> list[str]:
        """
        Generate queries for agentic re-retrieval based on gaps.
        
        Uses missing_information and insufficient sub-claims.
        """
        queries = []
        
        # From missing information
        for info in result.missing_information:
            if isinstance(info, str) and len(info) > 10:
                queries.append(info)
        
        # From insufficient sub-claims
        for subclaim in result.sub_claims:
            if subclaim.verdict == Verdict.INSUFFICIENT:
                queries.append(subclaim.claim)
        
        return queries[:3]  # Limit to avoid too many queries


def create_grounded_generator(
    api_key: Optional[str] = None,
    config: Optional[AdvancedRAGConfig] = None,
) -> GroundedGenerator:
    """Factory function to create GroundedGenerator."""
    client = Groq(api_key=api_key) if api_key else None
    return GroundedGenerator(client, config or DEFAULT_CONFIG)
