import pathway as pw
from sentence_transformers import SentenceTransformer
import os
from typing import Any

# Define Embedding Model Globally
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
GLOBAL_EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
CHUNK_SIZE = 2048
OVERLAP = 256

class NovelIngestionPipeline:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.model = GLOBAL_EMBEDDING_MODEL

    @staticmethod
    def _split_text_sliding_window(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
        words = text.split()
        n_words = len(words)
        chunks = []
        step = chunk_size - overlap
        for i in range(0, n_words, step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            relative_position = round(i / n_words, 3) if n_words > 0 else 0.0
            chunks.append((chunk_text, relative_position))
        return chunks

    @pw.udf
    def process_document(text: str) -> list[tuple[str, float]]:
        return NovelIngestionPipeline._split_text_sliding_window(text)

    @pw.udf
    def embed_text(text: str) -> list[float]:
        return GLOBAL_EMBEDDING_MODEL.encode(text, show_progress_bar=False).tolist()

    @staticmethod
    @pw.udf
    def get_metadata(filepath: Any) -> dict:
        path_str = str(filepath)
        if path_str.startswith('"') and path_str.endswith('"'):
            path_str = path_str[1:-1]
        filename = os.path.basename(str(path_str))
        return {'filename': filename}


    @staticmethod
    @pw.udf
    def get_book_name(filepath: Any) -> str:
        """
        Extracts clean filename (e.g., 'Harry_Potter') without extension.
        """
        path_str = str(filepath)
        # Remove JSON quotes if present
        if path_str.startswith('"') and path_str.endswith('"'):
            path_str = path_str[1:-1]

        # Get filename (book.txt)
        filename = os.path.basename(path_str)

        # Split extension (book, .txt) and return just the name
        name_only, _ = os.path.splitext(filename)
        return name_only

    def run_indexing(self):
        print(f"--- Ingesting from {self.data_dir} ---")
        # Read files
        files = pw.io.fs.read(self.data_dir, format="plaintext", mode="static", with_metadata=True)

        # Extract metadata
        documents = files.select(
            text=pw.this.data,
            # meta=NovelIngestionPipeline.get_metadata(pw.this._metadata["path"])
            book_name=NovelIngestionPipeline.get_book_name(pw.this._metadata["path"])
        )

        # Chunking
        chunks = documents.select(
            # metadata=pw.this.meta,
            pw.this.book_name,
            chunk_data=NovelIngestionPipeline.process_document(pw.this.text)
        ).flatten(pw.this.chunk_data)

        # Formatting
        structured_chunks = chunks.select(
            # book_name=pw.this.metadata["filename"],
            pw.this.book_name,
            chunk_text=pw.this.chunk_data[0],
            relative_position=pw.this.chunk_data[1]
        )

        # Embedding
        embedded_chunks = structured_chunks.select(
            pw.this.book_name,
            pw.this.chunk_text,
            pw.this.relative_position,
            vector=NovelIngestionPipeline.embed_text(pw.this.chunk_text)
        )

        return pw.debug.table_to_pandas(embedded_chunks)