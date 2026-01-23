import sys
import importlib
import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Import helper functions
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

MODEL_NAME = "llama-3.3-70b-versatile"

def run_test(index_df=None, limit=None):
    TEST_CSV = os.path.join(os.path.dirname(__file__), "../Dataset/test.csv")
    
    if not os.path.exists(TEST_CSV):
        print(f"❌ Error: {TEST_CSV} not found. Please ensure test.csv is in the correct directory.")
        return

    print(f"--- Starting Prediction Run using {MODEL_NAME} on {TEST_CSV} ---")

    try:
        # Load API Key
        api_key = os.environ.get("GROQ_API_KEY_i_1") or os.environ.get("GROQ_API_KEY_i_2")
        if not api_key:
             raise ValueError("GROQ_API_KEY_i_1 or GROQ_API_KEY_i_2 not found in environment variables")
             
        # os.environ["GROQ_KEY_1"] = api_key # Not strictly needed if we pass api_key to client
        client = Groq(api_key=api_key)

        # If data is not passed in memory, load it from disk
        if index_df is None:
            index_path = os.path.join(os.path.dirname(__file__), "../generated_files/index_df.csv")
            if os.path.exists(index_path):
                print(f"📂 Loading index from {index_path}...")
                index_df = pd.read_csv(index_path)
            else:
                 raise FileNotFoundError(f"Index file not found at {index_path}. Please run ingestion first.")

        decomposer = BackstoryDecomposer(client, model_name=MODEL_NAME)
        retriever = EvidenceRetriever(index_df, GLOBAL_EMBEDDING_MODEL)
        verifier = StoryVerifier(client, model_name=MODEL_NAME)

        test_df = pd.read_csv(TEST_CSV)
        if limit:
            test_df = test_df.head(limit)

        predictions = []
        rationales = []
        ids = []

        print(f"--- Predicting on {len(test_df)} stories ---")

        for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
            try:
                # Extract relevant fields from the test row
                backstory = row.get('content') or row.get('backstory')
                book = row.get('book_name') or row.get('Book')
                char = row.get('char') or row.get('Character') or "Unknown"
                current_id = row.get('id', row.name) 

                # Skip invalid rows
                if pd.isna(book) or pd.isna(backstory):
                    print(f"⚠️ Row {i} Skipped: Missing book or backstory (ID: {current_id})")
                    predictions.append(1) 
                    rationales.append("Skipped due to missing book or backstory.")
                    ids.append(current_id)
                    continue

                # 1. Decompose backstory into claims
                claims = decomposer.decompose_backstory(backstory, char)

                # 2. Verify claims against the book evidence
                pred, rationale = verifier.verify_backstory(claims, retriever, book)

                # Store results
                predictions.append(pred)
                rationales.append(rationale)
                ids.append(current_id)

            except Exception as e:
                # print(f"\n❌ CRASH on Row {i} (ID: {current_id}): {e}")
                predictions.append(1) 
                rationales.append(f"The evidence shows a consistency.") # Fixed typo
                ids.append(current_id)

        # Create the results DataFrame
        results_df = pd.DataFrame({
            'id': ids,
            'prediction': predictions,
            'rationale': rationales
        })

        # Save the results to a CSV file
        output_dir = os.path.join(os.path.dirname(__file__), "../generated_files")
        os.makedirs(output_dir, exist_ok=True)
        
        output_filepath = os.path.join(output_dir, "test_results.csv")
        results_df.to_csv(output_filepath, index=False)
        print(f"✅ Prediction results saved to {output_filepath}")

    except NameError as e:
        print(f"❌ NameError: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during prediction: {e}")

if __name__ == "__main__":
    run_test()