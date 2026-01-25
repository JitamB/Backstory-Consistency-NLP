import os
import sys
from dotenv import load_dotenv

# Ensure helper_functions can be found
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Load environment variables
load_dotenv()

from run_code.run_ingestion import run_ingestion
from run_code.run_validation import run_validation
from run_code.run_test import run_test
import pandas as pd
def main():
    print("🚀 Starting End-to-End Pipeline...")
    
    # Check for API Keys
    if not os.environ.get("GROQ_API_KEY_i_1") and not os.environ.get("GROQ_API_KEY_i_2"):
        print("❌ Warning: GROQ_API_KEY_i_1 not found in environment variables. Pipeline might fail during LLM calls.")

    # 1. Run Ingestion (returns index_df)
    try:
        index_df = run_ingestion()
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return

    # 2. Run Validation (uses index_df from memory)
    print("\n--- Running Validation ---")
    index_df = pd.read_csv("./generated_files/index_df.csv")
    run_validation(index_df, limit=2)

    # 3. Run Test (uses index_df from memory)
    print("\n--- Running Test ---")
    run_test(index_df, limit=2)

    print("\n✅ Pipeline Finished Successfully!")

if __name__ == "__main__":
    main()
