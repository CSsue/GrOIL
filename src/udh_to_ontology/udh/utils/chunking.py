import os
from transformers import AutoTokenizer
from dotenv import load_dotenv

def chunk_text_by_tokens(model: str = None ,text: str = None, chunk_size: int = 600, overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks based on Qwen's token count."""

    if not model:
        load_dotenv()
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")

    TOKENIZER = AutoTokenizer.from_pretrained(model_name)

    # Encode the text into Qwen's specific token IDs
    tokens = TOKENIZER.encode(text)

    chunks = []
    # Ensure step is at least 1 to prevent infinite loops
    step = max(1, chunk_size - overlap)

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        # Decode the token IDs back into a string chunk
        chunks.append(TOKENIZER.decode(chunk_tokens))

    return chunks