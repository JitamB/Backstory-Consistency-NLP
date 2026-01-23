from sentence_transformers import CrossEncoder, util
import torch
import numpy as np
import pandas as pd

RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

class EvidenceRetriever:
    def __init__(self, index_df, embedding_model):
        # 1. Clean Index
        self.index = index_df.reset_index(drop=True)
        self.index['book_name'] = self.index['book_name'].fillna('').astype(str)

        self.embedder = embedding_model
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

        # --- CRITICAL FIX: FORCE FLOAT32 ---
        # Parse vectors if they are strings (from CSV load)
        if isinstance(self.index['vector'].iloc[0], str):
             import ast
             self.index['vector'] = self.index['vector'].apply(ast.literal_eval)

        # Convert Pandas objects (float64) to float32 numpy array for PyTorch compatibility
        raw_vectors = np.stack(self.index['vector'].values)
        self.vector_stack = raw_vectors.astype(np.float32)

    def search(self, claim_data, book_title):
        if not book_title or pd.isna(book_title):
            return []

        # 1. Scope by book
        book_mask = self.index['book_name'].str.contains(str(book_title), case=False, regex=False, na=False)
        book_idx = self.index[book_mask].index

        if len(book_idx) == 0:
            return []

        # Get vectors for this book (already float32 from init)
        book_vectors = self.vector_stack[book_idx]
        book_chunks = self.index.loc[book_idx].reset_index(drop=True)

        # 2. Multi-query search
        candidate_indices = set()
        for query in claim_data['queries']:
            # Ensure query vector is also float32
            query_vec = self.embedder.encode(query).astype(np.float32)

            scores = util.cos_sim(query_vec, book_vectors)[0]

            if len(book_vectors) == 0: continue

            top_k = torch.topk(scores, k=min(15, len(book_vectors))).indices.tolist()
            candidate_indices.update(top_k)

        if not candidate_indices:
            return []

        # 3. Select Candidates
        candidates = book_chunks.iloc[list(candidate_indices)].copy()

        # 4. Rerank
        pred_pairs = [(claim_data['text'], row['chunk_text']) for _, row in candidates.iterrows()]
        candidates['score'] = self.reranker.predict(pred_pairs)

        # 5. Temporal Boost
        if claim_data.get('type') == 'TEMPORAL' and 'early' in claim_data['text']:
             candidates['score'] += np.where(candidates['relative_position'] < 0.2, 1.0, 0.0)

        return candidates.sort_values(by='score', ascending=False).head(5)[['chunk_text', 'score', 'relative_position']].to_dict('records')