"""
Configuration file for Legal Chatbot application.
Contains all configurable variables and settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration settings."""
    
    # OCR Settings
    OCR_BATCH_SIZE: int = 150  # Number of parallel OCR calls per batch (high for speed)
    OCR_DPI: int = 100  # DPI for PDF to image conversion (lower = faster, still readable)
    OCR_MAX_TOKENS: int = 4000  # Max tokens for OCR response
    OCR_TEMPERATURE: float = 0.1  # Temperature for OCR (low for accuracy)
    OCR_TIMEOUT: int = 180  # Timeout in seconds for OCR API calls
    OCR_MAX_CONCURRENT: int = 150  # Maximum concurrent API requests
    OCR_IMAGE_QUALITY: int = 70  # JPEG quality (lower = smaller = faster)
    OCR_MODEL: str = os.getenv("OCR_MODEL", "usf1-mini")  # USF vision model for OCR only
    
    # Data Extraction Output
    DATA_EXTRACTED_FOLDER: str = "data_extracted"  # Folder to save extracted text files
    
    # OpenAI LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5")
    # Structured JSON during ingestion; gpt-5 often returns empty content for strict JSON L2 prompts
    CHUNKING_LLM_MODEL: str = os.getenv("CHUNKING_LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "600"))
    # GPT-5 is slow; high concurrency causes 504/timeouts
    LLM_MAX_CONCURRENT: int = int(os.getenv("LLM_MAX_CONCURRENT", "8"))
    L2_MAX_OUTPUT_TOKENS: int = int(os.getenv("L2_MAX_OUTPUT_TOKENS", "2500"))
    
    # OpenAI Embeddings Settings
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    EMBED_DIMENSIONS: int = int(os.getenv("EMBED_DIMENSIONS", "3072"))
    EMBED_TIMEOUT: int = int(os.getenv("EMBED_TIMEOUT", "60"))
    
    # RAG Settings
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200
    RAG_TOP_K: int = 5
    
    # Hierarchical Chunking Settings
    LAYER1_CHUNK_SIZE: int = 1000  # Token size for Layer 1 chunks (reduced from 1500)
    LAYER_CHUNK_SIZE: int = 1000  # Token size for all layer chunks (reduced from 1500)
    SUMMARY_TARGET_SIZE: int = 400  # Target size for summaries (reduced from 500)
    LAYER_THRESHOLD: int = 1000  # Token threshold to trigger next layer (reduced from 1500)
    MAX_HIERARCHY_LAYERS: int = int(os.getenv("MAX_HIERARCHY_LAYERS", "8"))
    CHUNK_BATCH_SIZE: int = 5  # Number of chunks to process in parallel for LLM calls (legacy)
    
    # Queue-Based Parallel Processing Settings
    LLM_QUEUE_ENABLED: bool = True  # Enable queue-based parallel processing
    LLM_RETRY_MAX: int = 4  # Max retries per request
    LLM_RETRY_BASE_DELAY: float = 3.0  # Base delay for exponential backoff
    
    # USF API — OCR/vision only
    USF_API_URL: str = os.getenv("USF_API_URL", "https://api.us.inc/usf/v1/hiring/chat/completions")
    USF_API_KEY: str = os.getenv("USF_API_KEY", "")
    
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    
    # MongoDB Settings
    MONGO_DB_URI: str = os.getenv("MONGO_DB_URI", "mongodb+srv://LOVJEET:LOVJEETMONGO@cluster0.zpzj90m.mongodb.net/?retryWrites=true&w=majority")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "LegalChatBot")
    MONGO_CHAT_HISTORY_COLLECTION: str = os.getenv("MONGO_CHAT_HISTORY_COLLECTION", "chat_history")
    MONGO_INSTANCES_COLLECTION: str = os.getenv("MONGO_INSTANCES_COLLECTION", "instances")
    
    # Document Processing
    MAX_DOCUMENT_LENGTH: int = 50000  # Max characters for document processing
    MAX_COMPARISON_LENGTH: int = 30000  # Max characters for comparison
    
    # Hybrid Retrieval Settings
    HYBRID_LLM_WEIGHT: float = float(os.getenv("HYBRID_LLM_WEIGHT", "0.6"))
    HYBRID_VECTOR_WEIGHT: float = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.3"))
    HYBRID_IMPORTANCE_WEIGHT: float = float(os.getenv("HYBRID_IMPORTANCE_WEIGHT", "0.1"))
    HYBRID_VECTOR_TOP_K: int = int(os.getenv("HYBRID_VECTOR_TOP_K", "5"))
    HYBRID_MIN_VECTOR_SCORE: float = float(os.getenv("HYBRID_MIN_VECTOR_SCORE", "0.3"))
    # Above this clause count, skip LLM clause list and use vector-first retrieval
    HYBRID_LLM_CLAUSE_THRESHOLD: int = int(os.getenv("HYBRID_LLM_CLAUSE_THRESHOLD", "1000"))
    HYBRID_MEDIUM_THRESHOLD: int = int(os.getenv("HYBRID_MEDIUM_THRESHOLD", "100"))
    RETRIEVAL_LLM_MODEL: str = os.getenv("RETRIEVAL_LLM_MODEL", "gpt-4o-mini")
    # Fast, reliable model for greetings, memory, and meta chat (avoid gpt-5 empty replies)
    CONVERSATIONAL_LLM_MODEL: str = os.getenv(
        "CONVERSATIONAL_LLM_MODEL",
        os.getenv("RETRIEVAL_LLM_MODEL", "gpt-4o-mini"),
    )
    HYBRID_VECTOR_TOP_K_LARGE: int = int(os.getenv("HYBRID_VECTOR_TOP_K_LARGE", "10"))
    HYBRID_LLM_WEIGHT_LARGE: float = float(os.getenv("HYBRID_LLM_WEIGHT_LARGE", "0.0"))
    HYBRID_VECTOR_WEIGHT_LARGE: float = float(os.getenv("HYBRID_VECTOR_WEIGHT_LARGE", "0.85"))
    HYBRID_IMPORTANCE_WEIGHT_LARGE: float = float(os.getenv("HYBRID_IMPORTANCE_WEIGHT_LARGE", "0.15"))
    CLAUSE_INDEX_MONGO_THRESHOLD: int = int(os.getenv("CLAUSE_INDEX_MONGO_THRESHOLD", "500"))
    MONGO_CLAUSE_INDEX_COLLECTION: str = os.getenv("MONGO_CLAUSE_INDEX_COLLECTION", "clause_indices")
    QDRANT_SCROLL_PAGE_SIZE: int = int(os.getenv("QDRANT_SCROLL_PAGE_SIZE", "500"))


config = Config()
