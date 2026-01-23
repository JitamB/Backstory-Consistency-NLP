import sys
import importlib
import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

# Add the parent directory to sys.path to allow importing helper_functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

MODEL_NAME = "llama-3.3-70b-versatile"

def run_validation(index_df=None, limit=None):
    # 2. RELOAD MODULES (Updates code without restarting kernel)
    # modules_to_reload = ['query_generator', 'evidence_retrieval', 'verification', 'validator']
    # for m in modules_to_reload:
    #     if m in sys.modules:
    #         del sys.modules[m]

    try:
        import helper_functions.validator as validator
        importlib.reload(validator)
    except ImportError:
         import validator
         importlib.reload(validator)

    # 3. RUN VALIDATION
    try:
        api_key = os.environ.get("GROQ_API_KEY_i_1")
        if not api_key:
            raise ValueError("GROQ_API_KEY_i_1 not found in environment variables")
            
        client = Groq(api_key=api_key)
        print(f"🚀 Starting Validation Run using {MODEL_NAME}...")

        # If data is not passed in memory, load it from disk
        if index_df is None:
            index_path = os.path.join(os.path.dirname(__file__), "../generated_files/index_df.csv")
            if os.path.exists(index_path):
                print(f"📂 Loading index from {index_path}...")
                index_df = pd.read_csv(index_path)
            else:
                 raise FileNotFoundError(f"Index file not found at {index_path}. Please run ingestion first.")

        # Passing index_df
        validation_results_df = validator.run_validation(client, MODEL_NAME, index_df=index_df, limit=limit)

        output_dir = os.path.join(os.path.dirname(__file__), "../generated_files")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "validation_results.csv")
        validation_results_df.to_csv(output_path, index=False)
        print(f"✅ Validation results saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_validation()