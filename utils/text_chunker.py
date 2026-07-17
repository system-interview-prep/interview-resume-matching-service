from typing import List

def get_chunks(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Split text into overlapping sliding-window word chunks.
    
    Args:
        text: Raw text to split.
        chunk_size: Target word count per chunk.
        overlap: Number of overlapping words between adjacent chunks.
        
    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []
        
    words = text.strip().split()
    if len(words) <= chunk_size:
        return [text.strip()]
        
    chunks = []
    i = 0
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size // 2  # Fallback if overlap is larger than or equal to chunk_size
        
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) > 0:
            chunks.append(" ".join(chunk_words))
        i += step
        
    return chunks
