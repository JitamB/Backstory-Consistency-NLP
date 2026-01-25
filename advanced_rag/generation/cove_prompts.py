"""
Chain-of-Verification (CoVe) Prompt Templates

Implements structured prompting for near-zero hallucination verification.
Every claim requires explicit evidence citation or is marked unfounded.
"""

COVE_VERIFICATION_TEMPLATE = """You are a fact-checking system. Your goal is ZERO hallucinations.

## CLAIM TO VERIFY
{claim}

## EVIDENCE PASSAGES (with IDs like [e1], [e2], etc.)
{evidence_with_ids}

## VERIFICATION PROTOCOL (Follow Exactly)

### Step 1: Extract Atomic Sub-Claims
Break the claim into atomic facts that can be individually verified.

### Step 2: For Each Sub-Claim, Find Supporting Evidence
- Quote the EXACT TEXT from evidence that supports or contradicts
- If no evidence exists, state "NO EVIDENCE FOUND"
- Cite using the passage ID exactly as shown: [e1], [e2], etc.

### Step 3: Check for Contradictions
- Does any evidence DIRECTLY contradict the claim?
- Is the contradiction EXPLICIT or merely implied?

### Step 4: Confidence Assessment
- Rate your confidence: HIGH (>90%), MEDIUM (60-90%), LOW (<60%)
- If evidence supports the claim, verdict should be CONSISTENT
- If evidence contradicts the claim, verdict should be CONTRADICTION

### Step 5: Final Verdict
Provide your structured verdict. Use CONSISTENT if evidence supports, CONTRADICTION if evidence contradicts, INSUFFICIENT_EVIDENCE only if truly no relevant evidence found.

## CRITICAL RULES
1. NEVER infer beyond explicit evidence
2. Citations MUST use the exact IDs shown: [e1], [e2], [e3], etc.
3. If claim is supported by evidence, verdict is CONSISTENT (not INSUFFICIENT)
4. Quote evidence verbatim when possible
"""

CLAIM_DECOMPOSITION_TEMPLATE = """Analyze this backstory for character "{character_name}":
"{backstory}"

Extract 3-5 atomic, verifiable claims with these categories:
- TEMPORAL: Events with time/sequence (birth, death, when things happened)
- RELATIONSHIP: Who is related to whom, how
- LOCATION: Where things happened or where someone lived
- TRAIT: Character personality, abilities, appearance
- EVENT: Specific plot events (battles, marriages, etc.)

For EACH claim, provide 3 search queries:
1. Semantic query: Captures meaning for vector search
2. Keyword query: Exact names/terms for BM25
3. Negation query: Searches for contradicting evidence

Also estimate the expected narrative position (0.0=start to 1.0=end) where evidence should be found.
"""

SELF_CONSISTENCY_TEMPLATE = """You previously verified a claim. Now verify your own verification.

## ORIGINAL CLAIM
{claim}

## YOUR PREVIOUS VERDICT
{previous_verdict}

## EVIDENCE (same as before)
{evidence}

## SELF-CHECK QUESTIONS
1. Did I quote evidence accurately?
2. Did I miss any relevant passages?
3. Is my confidence level appropriate?
4. Are there any contradictions I overlooked?
5. Is my verdict justified by the evidence?

## FINAL DECISION
Confirm or revise your verdict with justification.
"""

TEMPORAL_CONSISTENCY_CHECK = """Check if evidence is temporally consistent with claim.

## CLAIM
{claim}
Expected narrative position: {expected_position}

## EVIDENCE
{evidence}
Evidence narrative position: {evidence_position}

## TEMPORAL CHECK
1. Is the evidence from an appropriate point in the story?
2. Could story events have changed what's described in the evidence?
3. Is this evidence still valid for the claim's context?

Respond with: CONSISTENT, PARTIALLY_CONSISTENT, or INCONSISTENT
And explain why.
"""
