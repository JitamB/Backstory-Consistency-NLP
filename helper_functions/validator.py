import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import os
import sys

# Import modules
# Import modules
try:
    from helper_functions.query_generator import BackstoryDecomposer
    from helper_functions.evidence_retrieval import EvidenceRetriever
    from helper_functions.verification import StoryVerifier
    from helper_functions.data_ingestion import GLOBAL_EMBEDDING_MODEL
except ImportError:
    from query_generator import BackstoryDecomposer
    from evidence_retrieval import EvidenceRetriever
    from verification import StoryVerifier
    from data_ingestion import GLOBAL_EMBEDDING_MODEL

TRAIN_CSV = "./Dataset/train.csv"

def map_label(val):
    """Converts string labels to integers (0/1) robustly."""
    if pd.isna(val): return 1 # Default to Consistent if missing

    # If already number
    if isinstance(val, (int, float)):
        return int(val)

    # If string
    s = str(val).lower().strip()
    if "contradict" in s: return 0
    if "fake" in s: return 0
    if "consistent" in s: return 1
    if "true" in s: return 1

    return 1 # Default fallback

def run_validation(client, model_name, index_df, limit=None):
    if not os.path.exists(TRAIN_CSV):
        print("❌ Please upload train.csv")
        return pd.DataFrame() # Return empty DataFrame on error

    if index_df is None or index_df.empty:
        print("❌ CRITICAL ERROR: Passed index_df is empty!")
        return pd.DataFrame() # Return empty DataFrame on error

    print(f"--- 2. Initializing Pipeline with Model: {model_name} ---")

    decomposer = BackstoryDecomposer(client, model_name=model_name)
    retriever = EvidenceRetriever(index_df, GLOBAL_EMBEDDING_MODEL)
    verifier = StoryVerifier(client, model_name=model_name)

    df = pd.read_csv(TRAIN_CSV)
    if limit: df = df.head(limit)

    preds, truths = [], []
    ids = [] # To store the IDs
    rationales = []

    print(f"--- 3. Validating {len(df)} stories ---")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # --- INPUT SANITIZATION ---
            backstory = row.get('content') or row.get('backstory')
            book = row.get('book_name') or row.get('Book')
            char = row.get('char') or row.get('Character') or "Unknown"
            # Get ID, prefer 'id' column, otherwise use row index
            current_id = row.get('id', row.name)

            # --- LABEL HANDLING (FIXED) ---
            raw_label = None
            if 'label' in row: raw_label = row['label']
            elif 'Label' in row: raw_label = row['Label']
            elif 'verdict' in row: raw_label = row['verdict']

            truth = map_label(raw_label)

            # Skip invalid rows
            if pd.isna(book) or pd.isna(backstory):
                print(f"⚠️ Row {i} Skipped: Missing book or backstory")
                preds.append(0)
                truths.append(truth)
                ids.append(current_id) # Append ID even for skipped rows
                continue

            # 1. Decompose
            claims = decomposer.decompose_backstory(backstory, char)

            # 2. Verify
            pred, rationale = verifier.verify_backstory(claims, retriever, book)

            preds.append(pred)
            truths.append(truth)
            ids.append(current_id) # Append ID
            rationales.append(rationale)

        except Exception as e:
            print(f"\n❌ CRASH on Row {i}: {e}")
            preds.append(1)
            truths.append(1)
            ids.append(current_id) # Append ID for crashed rows

    print("\nResults:")
    try:
        print(f"Accuracy: {accuracy_score(truths, preds)}")
        print(classification_report(truths, preds, target_names=["Contradiction (0)", "Consistent (1)"]))
    except Exception as e:
        print(f"Could not calculate metrics: {e}")
        print(f"Preds sample: {preds[:5]}")
        print(f"Truths sample: {truths[:5]}")

    # Create and return the results DataFrame
    results_df = pd.DataFrame({
        'id': ids,
        'truth': truths,
        'prediction': preds,
        'rationale': rationales
    })
    return results_df # Modify return value

if __name__ == "__main__":
    pass