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


## Project Structure

```
Backstory-Consistency-NLP/
├── main.py                     # Entry point to run the full pipeline
├── requirements.txt            # Project dependencies
├── .env                        # Environment variables (API Keys)
├── helper_functions/           # Reusable logic modules
│   ├── data_ingestion.py
│   ├── evidence_retrieval.py
│   ├── query_generator.py
│   ├── validator.py
│   └── verification.py
├── run_code/                   # Standalone scripts for individual steps
│   ├── run_ingestion.py
│   ├── run_test.py
│   └── run_validation.py
└── generated_files/            # Output directory for CSVs (for testing purpose only 2 input is taken)
```


## How to Run

### Prerequisites

1. Python 3.11 or higher
2. Groq API Key

Clone the repository in your local machine and navigate to the project directory. (Creating a local environment is recommended)
### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the root directory and add your API keys:

```bash
GROQ_API_KEY_i_1=your_api_key_1
GROQ_API_KEY_i_2=your_api_key_2
```

### Run Pipeline (Two ways)

#### 1. Run Full Pipeline (Recommended)
Runs Ingestion → Validation → Testing sequentially in one step.
```bash
python main.py
```
After the run is complete, all the output files will be saved in `generated_files/` directory.

#### 2. Run Individual Steps (For Debugging)
You can run any step in isolation. They will automatically load/save data to `generated_files/`.

**Ingestion:**
```bash
python run_code/run_ingestion.py
```

**Validation:**
```bash
python run_code/run_validation.py
```

**Testing:**
```bash
python run_code/run_test.py
```