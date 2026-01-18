# Backstory-Consistency-NLP

## Solution Approach

The solution is a Retrieval-Augmented Generation (RAG) pipeline designed to verify the consistency of character backstories against a corpus of books. It consists of four main stages:

### 1. Data Ingestion
- **Tools**: Pathway, SentenceTransformers (`all-MiniLM-L6-v2`).
- **Process**:
    - Reads text files from the `Dataset/Books` directory.
    - extract book names from filenames.
    - Splits text into chunks using a sliding window approach (2048 token size, 256 overlap) to maintain context.
    - Embeds chunks using the `all-MiniLM-L6-v2` model.
    - Stores embeddings in a Pathway table for efficient retrieval.

### 2. Query Generation (Backstory Decomposition)
- **Tools**: Groq API (LLM).
- **Process**:
    - Takes a user-provided character backstory.
    - Uses an LLM to decompose the backstory into atomic, verifiable claims (e.g., Temporal, Relationship, Location, Trait).
    - For each claim, three types of search queries are generated:
        - **Keyword search**: For broad matching.
        - **Descriptive search**: For semantic matching.
        - **Anti-evidence search**: Specifically designed to find contradictions.

### 3. Evidence Retrieval
- **Tools**: SentenceTransformers (`CrossEncoder/ms-marco-MiniLM-L-6-v2`), PyTorch, Pandas.
- **Process**:
    - **Scoping**: Filters the search to the specific book mentioned in the query.
    - **Vector Search**: Retrieves top candidates based on cosine similarity of embeddings.
    - **Re-ranking**: A Cross-Encoder model scores the relevance of the retrieved chunks against the claim to refine the results.
    - **Temporal Boosting**: Applies a heuristic boost to scores if the claim involves "early" life events and the chunk appears early in the book (relative position < 0.2).

### 4. Verification
- **Tools**: Groq API (LLM).
- **Process**:
    - The LLM acts as a strict fact-checker.
    - It evaluates the claim against the retrieved evidence.
    - Outputs a JSON verdict: `SUPPORT`, `CONTRADICT`, or `NEUTRAL`, along with a confidence score and rationale.
    - A contradiction is flagged if the verdict is `CONTRADICT` with high confidence (>= 0.5).