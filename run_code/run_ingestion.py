import sys
import os
import pandas as pd

# Add the parent directory to sys.path to allow importing helper_functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Clear cached modules to ensure updates apply
if 'data_ingestion' in sys.modules: del sys.modules['data_ingestion']
try:
    from helper_functions.data_ingestion import NovelIngestionPipeline
except ImportError:
    # Fallback if helper_functions is in python path differently or running from root
    from data_ingestion import NovelIngestionPipeline

def run_ingestion():
    # 2. Define your path
    # Use absolute path or relative to project root
    BOOKS_DIR = os.path.join(os.path.dirname(__file__), "../Dataset/Books")
    
    # 3. Run Ingestion
    print("📚 Starting Ingestion...")
    ingestion_pipeline = NovelIngestionPipeline(BOOKS_DIR)
    index_df = ingestion_pipeline.run_indexing()
    
    print(f"✅ Ingestion Complete. Index contains {len(index_df)} chunks.")
    print(index_df.head(2))
    
    # Ensure generated_files directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "../generated_files")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "index_df.csv")
    index_df.to_csv(output_path, index=False)
    print(f"✅ Index saved to {output_path}")
    
    return index_df

if __name__ == "__main__":
    run_ingestion()
