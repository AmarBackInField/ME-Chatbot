import os
import sys
import tempfile
import tiktoken
import time
import asyncio
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Initialize tiktoken encoder for accurate token counting
tiktoken_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens accurately using tiktoken."""
    return len(tiktoken_encoder.encode(text))

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from RagService.RagService import RAGService
from DataExtraction.image_ocr import ImageOCR
from utils.logger import get_logger
from agents.conversation_graph import ConversationGraph
from services.mongodb_service import get_mongodb_service
from RagService.hybrid_retrieval import HybridRetriever, create_hybrid_retriever

logger = get_logger("API")

# Initialize MongoDB Service
mongodb_service = None
try:
    mongodb_service = get_mongodb_service()
    logger.info("MongoDB service initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize MongoDB service: {e}")

app = FastAPI(
    title="Mechanical Engineering Document Analysis API",
    description="AI-powered mechanical engineering document analysis with specification extraction, safety information retrieval, and troubleshooting guidance",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from config import config

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USF_API_KEY = os.getenv("USF_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def get_collection_vector_size(collection_info) -> int:
    """Return vector dimension for a Qdrant collection."""
    params = collection_info.config.params
    vectors = params.vectors
    if hasattr(vectors, "size"):
        return vectors.size
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()), None)
        if first is not None and hasattr(first, "size"):
            return first.size
    return 0


def validate_collection_embedding_dims(collection_name: str):
    """Raise HTTP 409 if collection uses legacy embedding dimensions."""
    if rag_service is None:
        return
    collection_info = rag_service.qdrant_client.get_collection(collection_name)
    vector_size = get_collection_vector_size(collection_info)
    if vector_size and vector_size != config.EMBED_DIMENSIONS:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This session was indexed with an older embedding model "
                f"({vector_size} dimensions). Please re-upload the document for this session."
            ),
        )

# Initialize RAG Service
rag_service = None
try:
    rag_service = RAGService()
    print("RAG Service initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize RAG Service: {e}")

# Cache for HybridRetriever instances (one per collection)
hybrid_retriever_cache: Dict[str, HybridRetriever] = {}

def get_hybrid_retriever(collection_name: str) -> HybridRetriever:
    """Get or create a cached HybridRetriever for a collection."""
    if collection_name not in hybrid_retriever_cache:
        hybrid_retriever_cache[collection_name] = create_hybrid_retriever(
            collection_name=collection_name,
            qdrant_url=rag_service.qdrant_url if rag_service else None,
            qdrant_api_key=rag_service.qdrant_api_key if rag_service else None
        )
        logger.info(f"Created and cached HybridRetriever for collection: {collection_name}")
    return hybrid_retriever_cache[collection_name]




# Initialize Image OCR service
image_ocr = None
try:
    image_ocr = ImageOCR()
    logger.info("ImageOCR service initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize ImageOCR service: {e}")

# Initialize Conversation Graph
conversation_graph = None

try:
    conversation_graph = ConversationGraph()
    logger.info("Conversation graph initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize conversation graph: {e}")


async def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using Image OCR."""
    try:
        if image_ocr is not None:
            text = await image_ocr.extract_text_from_pdf(file_path)
            return text
        else:
            raise Exception("ImageOCR service not available")
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise Exception(f"Error reading PDF: {str(e)}")






@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Mechanical Engineering Document Analysis API",
        "version": "3.0.0",
        "endpoints": {
            "/chat": "POST - Conversational analysis of technical documents",
            "/upload-document": "POST - Upload technical document for analysis",
            "/session/{session_id}": "GET - Get session info",
            "/session/{session_id}/history": "GET - Get chat history",
            "/sessions": "GET - List all sessions",
            "/create-collection": "POST - Create RAG collection",
            "/collections": "GET - List collections",
            "/delete-collection/{collection_name}": "DELETE - Delete collection",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "api_configured": bool(OPENAI_API_KEY),
        "ocr_configured": bool(USF_API_KEY),
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "embed_dimensions": config.EMBED_DIMENSIONS,
        "rag_service_available": rag_service is not None,
    }


@app.post("/create-collection")
async def create_collection(collection_name: str = "main_collection"):
    """Create a new collection in the RAG system."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG Service not available")
    
    try:
        rag_service.create_collection(collection_name)
        return {"status": "success", "message": f"Collection '{collection_name}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")


@app.get("/collections")
async def list_collections():
    """List available collections in the RAG system."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG Service not available")
    
    try:
        collections = rag_service.qdrant_client.get_collections().collections
        return {
            "status": "success",
            "collections": [col.name for col in collections]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")


@app.delete("/delete-collection/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection from the RAG system. Use this to recreate collections with correct vector dimensions."""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG Service not available")
    
    try:
        rag_service.delete_collection(collection_name)
        return {"status": "success", "message": f"Collection '{collection_name}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


# ==================== CONVERSATIONAL CHAT ENDPOINT ====================

# Store document context for chat sessions (in-memory, for demo purposes)
document_sessions: Dict[str, Dict[str, str]] = {}


async def resolve_document_text(session_id: str, collection_name: str) -> str:
    """Load full document text from memory or MongoDB (by session or collection)."""
    for key in (session_id, collection_name):
        if key in document_sessions:
            text = document_sessions[key].get("document_text", "")
            if text and text.strip():
                logger.info(f"Resolved document_text from memory ({len(text)} chars, key={key})")
                return text

    if mongodb_service:
        for lookup_id, label in ((session_id, "session_id"), (None, "collection_name")):
            try:
                if label == "session_id":
                    instance = await mongodb_service.get_instance(session_id)
                else:
                    instance = await mongodb_service.get_instance_by_collection(collection_name)
                if instance:
                    text = instance.get("document_text", "") or ""
                    if text.strip():
                        logger.info(
                            f"Resolved document_text from MongoDB ({len(text)} chars, by {label})"
                        )
                        document_sessions[session_id] = {
                            "document_text": text,
                            "document_name": instance.get("document_name", "unknown"),
                            "template_text": instance.get("template_text"),
                            "collection_name": collection_name,
                        }
                        document_sessions[collection_name] = document_sessions[session_id]
                        return text
            except Exception as e:
                logger.warning(f"Could not load document_text by {label}: {e}")

    logger.warning(
        f"No document_text found for session_id={session_id}, collection_name={collection_name}"
    )
    return ""


@app.post("/upload-document")
async def upload_document(
    document: UploadFile = File(...),
    collection_name: str = Form(...)
):
    """
    Upload a document for conversational analysis.
    
    Flow:
    1. Extract text from PDF document
    2. Apply chunking to the extracted text
    3. Store chunks with embeddings in vector database (Qdrant)
    
    Parameters:
    - document: PDF file to upload
    - collection_name: Name of the collection where chunks will be stored
    
    Returns session_id to use for subsequent chat queries.
    """
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG Service not available")
    
    if not document.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await document.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Check if collection already exists BEFORE extraction
        logger.info(f"Checking if collection '{collection_name}' already exists...")
        try:
            collections = rag_service.qdrant_client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)
            
            if collection_exists:
                collection_info = rag_service.qdrant_client.get_collection(collection_name)
                points_count = collection_info.points_count
                vector_size = get_collection_vector_size(collection_info)

                if vector_size and vector_size != config.EMBED_DIMENSIONS:
                    logger.info(
                        f"Recreating collection '{collection_name}' "
                        f"(legacy {vector_size}d -> {config.EMBED_DIMENSIONS}d)"
                    )
                    rag_service.delete_collection(collection_name)
                    hybrid_retriever_cache.pop(collection_name, None)
                    collection_exists = False
                elif points_count > 0:
                    logger.warning(f"Collection '{collection_name}' already exists with {points_count} chunks")
                    raise HTTPException(
                        status_code=409,
                        detail=f"Collection '{collection_name}' already exists with {points_count} chunks. Please use a different collection name or delete the existing collection first."
                    )
                else:
                    logger.info(f"Collection '{collection_name}' exists but is empty - will proceed with ingestion")
            else:
                logger.info(f"Collection '{collection_name}' does not exist - will create it")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error checking collection: {e}")
            raise HTTPException(status_code=500, detail=f"Error checking collection: {str(e)}")
        
        # Step 1: Extract text from PDF
        logger.info(f"Step 1: Extracting text from '{document.filename}'")
        document_text = await extract_text_from_pdf(tmp_file_path)
        
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        logger.info(f"Extraction complete: {len(document_text)} characters, {count_tokens(document_text)} tokens")
        
        # Step 2 & 3: Chunk the text and store with embeddings in vector database
        logger.info(f"Step 2 & 3: Chunking and storing in collection '{collection_name}'")
        
        # Create collection if it doesn't exist
        if not collection_exists:
            try:
                rag_service.create_collection(collection_name)
                logger.info(f"Created new collection: '{collection_name}'")
            except Exception as e:
                logger.error(f"Error creating collection: {e}")
                raise HTTPException(status_code=500, detail=f"Error creating collection: {str(e)}")
        
        # Ingest document with hierarchical chunking and embeddings
        document_id = document.filename
        ingest_result = await rag_service.ingest_hierarchical_chunks(
            text=document_text,
            collection_name=collection_name,
            document_id=document_id
        )
        
        total_chunks = ingest_result.get("total_chunks", 0)
        num_layers = ingest_result.get("num_layers", 0)
        global_clause_index = ingest_result.get("global_clause_index") or {}
        clause_type_counts = ingest_result.get("clause_type_counts") or {}
        clause_index_entries = ingest_result.get(
            "clause_index_entries", len(global_clause_index)
        )
        
        logger.info(f"Ingestion complete: {total_chunks} chunks across {num_layers} layers stored in '{collection_name}'")
        
        # Generate a session_id for this upload
        session_id = f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store document text in session (in-memory) for backward compatibility
        session_data = {
            "document_text": document_text,
            "document_name": document.filename,
            "template_text": None,
            "collection_name": collection_name,
        }
        document_sessions[session_id] = session_data
        document_sessions[collection_name] = session_data
        
        # Store instance in MongoDB
        if mongodb_service:
            try:
                await mongodb_service.create_instance(
                    session_id=session_id,
                    document_name=document.filename,
                    document_text=document_text,
                    metadata={
                        "uploaded_at": datetime.now().isoformat(),
                        "collection_name": collection_name,
                        "total_chunks": total_chunks,
                        "num_layers": num_layers,
                        "clause_index_entries": clause_index_entries,
                    }
                )
                if (
                    global_clause_index
                    and clause_index_entries >= config.CLAUSE_INDEX_MONGO_THRESHOLD
                ):
                    await mongodb_service.save_clause_index(
                        collection_name=collection_name,
                        clause_index=global_clause_index,
                        clause_type_counts=clause_type_counts,
                        session_id=session_id,
                    )
                logger.info(f"Instance stored in MongoDB for session: {session_id}")
            except Exception as e:
                logger.warning(f"Could not store instance in MongoDB: {e}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "collection_name": collection_name,
            "document_name": document.filename,
            "ingestion_info": {
                "total_chunks": total_chunks,
                "num_layers": num_layers,
                "characters": len(document_text),
                "tokens": count_tokens(document_text),
                "clause_index_entries": clause_index_entries,
            },
            "message": f"Document '{document.filename}' uploaded successfully. {total_chunks} chunks stored in collection '{collection_name}'. You can now ask questions about it using session_id: {session_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    message: str = Form(...),
    collection_name: str = Form(...),
    use_rag: bool = Form(default=True),
    hybrid_mode: bool = Form(default=True)
):
    """
    Conversational endpoint for natural language queries about technical documents in a collection.
    
    Prerequisites:
    - Upload a document first using /upload-document endpoint to create a collection
    - Use the collection_name from the upload response
    
    Parameters:
    - session_id: Session identifier for conversation tracking
    - message: Your question or query about the technical document
    - collection_name: Name of the collection containing the document (required)
    - use_rag: Enable/disable RAG context retrieval (default: True)
    - hybrid_mode: Enable hybrid search (semantic + section-aware retrieval) (default: True)
    
    Example queries:
    - "What are the torque specifications?"
    - "What are the safety warnings?"
    - "How do I troubleshoot error E42?"
    - "What materials are specified?"
    - "What are the operating temperature limits?"
    """
    # Initialize latency tracking
    start_time = time.time()
    latency_metrics = {}
    
    if conversation_graph is None:
        raise HTTPException(status_code=503, detail="Conversation service not available")
    
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG Service not available")
    
    # Check if collection exists
    step_start = time.time()
    logger.info(f"🔍 Checking if collection '{collection_name}' exists")
    try:
        collections = rag_service.qdrant_client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)
        
        if not collection_exists:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' does not exist. Please upload a document first using /upload-document endpoint."
            )
        
        # Check if collection has data
        collection_info = rag_service.qdrant_client.get_collection(collection_name)
        points_count = collection_info.points_count
        
        if points_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' exists but is empty. Please upload a document first."
            )

        validate_collection_embedding_dims(collection_name)

        latency_metrics['collection_validation'] = time.time() - step_start
        logger.info(f"✅ Collection '{collection_name}' found with {points_count} chunks (⏱️ {latency_metrics['collection_validation']:.2f}s)")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking collection: {str(e)}")
    
    document_text = await resolve_document_text(session_id, collection_name)
    
    # OPTIMIZATION: Quick rule-based intent detection (saves 5.23s for 70% of queries)
    def quick_intent_check(query: str) -> Optional[str]:
        """Fast rule-based intent classification before LLM."""
        query_lower = query.lower()
        
        # Technical specs keywords
        if any(word in query_lower for word in ["specification", "spec", "tolerance", "dimension", "material", "torque", "pressure", "temperature"]):
            return "technical_specs"
        
        # Safety info keywords
        if any(word in query_lower for word in ["safety", "warning", "hazard", "limit", "caution", "danger"]):
            return "safety_info"
        
        # Troubleshooting keywords
        if any(word in query_lower for word in ["troubleshoot", "error", "fault", "maintenance", "repair", "diagnose", "fix"]):
            return "troubleshooting"
        
        # Design parameters keywords
        if any(word in query_lower for word in ["design", "calculation", "standard", "ISO", "ASME", "ANSI", "compliance"]):
            return "design_parameters"
        
        # General question keywords
        if any(word in query_lower for word in [
            "hello", "hi", "hey", "help", "assist", "what can you",
            "how can you", "how do you", "who are you", "who built", "candex",
        ]):
            if not any(word in query_lower for word in ["specification", "spec", "safety", "procedure", "section"]):
                return "general_question"
        
        return None  # Fall back to LLM classification
    
    # Try quick intent detection
    quick_intent = quick_intent_check(message)
    
    # OPTIMIZATION: Parallel execution of independent tasks
    logger.info("🚀 Starting parallel execution: Intent + RAG + History")
    parallel_start = time.time()
    
    # Define parallel tasks
    async def get_intent():
        """Task 1: Intent classification (5.23s or instant if quick match)."""
        if quick_intent:
            logger.info(f"⚡ Quick intent match: {quick_intent} (saved ~5s)")
            latency_metrics['intent_classification'] = 0.001  # Instant
            return quick_intent
        
        step_start = time.time()
        logger.info(f"🎯 [Parallel] Classifying intent with LLM: '{message[:50]}...'")
        intent = await conversation_graph._llm_classify_intent(message)
        latency_metrics['intent_classification'] = time.time() - step_start
        logger.info(f"✅ [Parallel] Intent classified: {intent} (⏱️ {latency_metrics['intent_classification']:.2f}s)")
        return intent
    
    async def get_rag_results():
        """Task 2: RAG retrieval (25.70s) - runs in parallel."""
        if not use_rag:
            return None, {}, "", False, {}
        
        try:
            rag_start = time.time()
            logger.info(f"🔀 [Parallel] Starting RAG retrieval: '{message[:50]}...'")
            
            if hybrid_mode:
                # Get cached HybridRetriever instance
                hybrid_retriever = get_hybrid_retriever(collection_name)
                
                # Conditional answer generation (skip for comparison to save time)
                should_generate_answer = quick_intent != "comparison"
                
                # Run hybrid retrieval
                hybrid_result = await hybrid_retriever.retrieve(
                    query=message,
                    top_k=5,
                    generate_answer=should_generate_answer
                )
                
                # Convert to expected format
                rag_results = []
                for chunk in hybrid_result.get("chunks", []):
                    rag_results.append({
                        "chunk_id": chunk.get("chunk_id"),
                        "layer": chunk.get("layer"),
                        "text": chunk.get("text"),
                        "short_summary": chunk.get("short_summary", ""),
                        "score": chunk.get("hybrid_score", 0),
                        "hybrid_score": chunk.get("hybrid_score", 0),
                        "llm_score": chunk.get("llm_score", 0),
                        "vector_score": chunk.get("vector_score", 0),
                        "clauses": chunk.get("clauses", []),
                        "risk_level": chunk.get("metadata", {}).get("risk_level", "unknown"),
                        "search_sources": ["llm", "vector"] if chunk.get("source") == "both" else [chunk.get("source", "unknown")],
                        "matching_clauses": chunk.get("matched_clause_ids", [])
                    })
                
                hybrid_answer = hybrid_result.get("answer", "")
                clause_index = hybrid_result.get("clause_index", {})
                hybrid_retrieval_meta = {
                    "retrieval_mode": hybrid_result.get("retrieval_mode", "hybrid"),
                    "clause_count": hybrid_result.get(
                        "clause_count",
                        hybrid_result.get("total_clauses_available", 0),
                    ),
                    "llm_clauses_selected": len(
                        hybrid_result.get("llm_selected_clauses", [])
                    ),
                }
                
                latency_metrics['rag_retrieval'] = time.time() - rag_start
                logger.info(f"✅ [Parallel] RAG completed (⏱️ {latency_metrics['rag_retrieval']:.2f}s)")
                logger.info(f"   LLM selected: {len(hybrid_result.get('llm_selected_clauses', []))} clauses")
                logger.info(f"   LLM chunks: {hybrid_result.get('llm_chunks_count', 0)}")
                logger.info(f"   Vector chunks: {hybrid_result.get('vector_chunks_count', 0)}")
                logger.info(f"   Final chunks: {len(rag_results)}")
                
                return rag_results, clause_index, hybrid_answer, True, hybrid_retrieval_meta
            else:
                rag_results = await rag_service.hierarchical_search(
                    query=message,
                    collection_name=collection_name,
                    top_k=3,
                    include_context=True
                )
                latency_metrics['rag_retrieval'] = time.time() - rag_start
                logger.info(f"✅ [Parallel] RAG completed (⏱️ {latency_metrics['rag_retrieval']:.2f}s)")
                return rag_results, {}, "", True, {}
        except Exception as e:
            logger.error(f"❌ [Parallel] RAG retrieval failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, {}, "", False, {}
    
    async def get_history():
        """Task 3: MongoDB history read (0.66s) - runs in parallel."""
        if not mongodb_service:
            return []
        
        try:
            step_start = time.time()
            logger.info(f"📚 [Parallel] Fetching conversation history")
            history = await mongodb_service.get_recent_messages(session_id, count=10)
            latency_metrics['mongodb_read'] = time.time() - step_start
            logger.info(f"✅ [Parallel] Retrieved {len(history)} messages (⏱️ {latency_metrics['mongodb_read']:.2f}s)")
            return history
        except Exception as e:
            logger.warning(f"Could not retrieve conversation history: {e}")
            return []
    
    # Execute all 3 tasks in parallel (OPTIMIZATION!)
    intent, rag_data, conversation_history = await asyncio.gather(
        get_intent(),
        get_rag_results(),
        get_history()
    )
    
    parallel_time = time.time() - parallel_start
    sequential_time = latency_metrics.get('intent_classification', 0) + latency_metrics.get('rag_retrieval', 0) + latency_metrics.get('mongodb_read', 0)
    time_saved = sequential_time - parallel_time
    logger.info(f"⚡ Parallel execution completed in {parallel_time:.2f}s (saved ~{time_saved:.2f}s vs sequential)")
    
    # Unpack RAG results
    rag_results, clause_index, hybrid_answer, rag_retrieved, hybrid_retrieval_meta = rag_data
    
    # Using existing data in collection
    doc_chars = 10000
    approx_tokens = 5000
    logger.info(f"Using existing data in collection '{collection_name}'")
    
    # Determine if RAG should be used based on intent
    rag_required_intents = ["technical_specs", "safety_info", "troubleshooting", "design_parameters"]
    should_use_rag = use_rag and (intent in rag_required_intents)
    
    rag_context = ""
    rag_used = False
    rag_decision_reason = ""
    
    if not use_rag:
        rag_decision_reason = "RAG disabled by user (use_rag=false)"
        logger.info(f"RAG Decision: SKIP - {rag_decision_reason}")
    elif intent not in rag_required_intents:
        rag_decision_reason = f"Intent '{intent}' does not require RAG retrieval"
        logger.info(f"RAG Decision: SKIP - {rag_decision_reason}")
        # For general questions, provide context about the available document
        document_text = f"Note: A technical document is available in collection '{collection_name}' with {points_count} chunks. If the user asks about the document, inform them you can analyze it for specifications, safety information, troubleshooting, or design parameters."
        # Discard RAG results if retrieved but not needed
        if rag_retrieved:
            logger.info(f"⚠️  RAG was retrieved in parallel but intent '{intent}' doesn't require it - discarding")
    elif rag_retrieved and rag_results:
        # Use the RAG results retrieved in parallel
        rag_decision_reason = f"Intent '{intent}' requires RAG retrieval"
        logger.info(f"RAG Decision: USE RAG - {rag_decision_reason}")
        logger.info(f"📊 RAG Results: {len(rag_results)} chunks retrieved")
        
        # Build RAG context
        rag_context_parts = []
        for i, result in enumerate(rag_results, 1):
            chunk_id = result.get("chunk_id", "unknown")
            layer = result.get("layer", "unknown")
            text = result.get("text", "")
            score = result.get("hybrid_score", result.get("score", 0))
            clauses = result.get("clauses", [])
            risk_level = result.get("risk_level", "unknown")
            search_sources = result.get("search_sources", ["semantic"])
            matching_clauses = result.get("matching_clauses", [])
            
            sources_str = "+".join(search_sources) if search_sources else "semantic"
            logger.info(f"  Chunk {i}: {chunk_id} | Layer: {layer} | Score: {score:.2f} | Sources: {sources_str} | Clauses: {len(clauses)}")
            
            context_header = f"[Context {i}: {chunk_id} (Layer {layer}, Risk: {risk_level}, Score: {score:.2f})]"
            
            if matching_clauses:
                clause_info = f"Matching Clauses: {len(matching_clauses)} found | Total Clauses: {len(clauses)}"
            else:
                clause_info = f"Clauses: {len(clauses)} identified" if clauses else "No clauses"
            
            rag_context_parts.append(f"{context_header}\n{clause_info}\n{text}")
            
            if "parent_chunks" in result and result["parent_chunks"]:
                parent_summaries = [p.get("short_summary", "") for p in result["parent_chunks"] if p.get("short_summary")]
                if parent_summaries:
                    rag_context_parts.append(f"[Parent Context]: {parent_summaries[0][:200]}...")
        
        rag_context = "\n\n".join(rag_context_parts)
        
        # Add hybrid LLM answer if available
        if hybrid_mode and hybrid_answer:
            rag_context = f"[HYBRID RETRIEVAL ANALYSIS]\n{hybrid_answer}\n\n[SUPPORTING DOCUMENT EXCERPTS]\n{rag_context}"
            logger.info(f"💬 Added hybrid LLM answer to RAG context ({len(hybrid_answer)} chars)")
        
        rag_used = True
        logger.info(f"✅ RAG Context Built: {len(rag_results)} results, {len(rag_context)} chars total")
        logger.info(f"📝 RAG Context Preview: {rag_context[:200]}...")
    else:
        logger.warning("⚠️  RAG retrieval failed or returned no results")
        rag_context = ""
        rag_used = False

    if not document_text.strip() and rag_used and rag_context:
        document_text = rag_context
        logger.info(f"Using RAG context as document_text fallback ({len(document_text)} chars)")

    try:
        logger.info(f"🤖 Calling conversation_graph.chat with:")
        logger.info(f"   - user_query length: {len(message)}")
        logger.info(f"   - document_text length: {len(document_text)}")
        logger.info(f"   - rag_context length: {len(rag_context)}")
        logger.info(f"   - rag_used: {rag_used}")
        
        # Conversation history already retrieved in parallel phase
        logger.info(f"Using conversation history from parallel fetch: {len(conversation_history)} messages")
        
        # Store user message in MongoDB
        step_start = time.time()
        if mongodb_service:
            try:
                await mongodb_service.add_chat_message(
                    session_id=session_id,
                    role="user",
                    content=message,
                    rag_used=rag_used,
                    metadata={"collection_name": collection_name}
                )
            except Exception as e:
                logger.warning(f"Could not store user message in MongoDB: {e}")
        latency_metrics['mongodb_write_user'] = time.time() - step_start
        
        # Conversation graph processing
        step_start = time.time()
        
        # Fast-path for general questions — conversational assistant (gpt-4o-mini, memory, CandexAI)
        if intent == "general_question" and not rag_used:
            logger.info("⚡ Fast-path: Conversational assistant (CandexAI)")
            doc_name = ""
            sess = document_sessions.get(collection_name) or document_sessions.get(session_id) or {}
            doc_name = sess.get("document_name", "")
            try:
                response_text = await conversation_graph.generate_conversational_response(
                    user_query=message,
                    conversation_history=conversation_history,
                    collection_name=collection_name,
                    document_available=points_count > 0,
                    document_name=doc_name,
                )
                result = {
                    "intent": intent,
                    "response": response_text,
                    "data": {"answer": response_text, "conversational": True},
                }
            except Exception as e:
                logger.warning(f"Conversational fast-path failed, falling back to graph: {e}")
                result = await conversation_graph.chat(
                    user_query=message,
                    document_text=document_text,
                    rag_context=rag_context,
                    session_id=session_id,
                    conversation_history=conversation_history,
                    clause_index=clause_index if 'clause_index' in locals() else {},
                    retrieval_metadata={
                        **(hybrid_retrieval_meta if 'hybrid_retrieval_meta' in locals() else {}),
                        "collection_name": collection_name,
                    },
                    collection_name=collection_name,
                    document_name=doc_name,
                )
        else:
            # Normal processing for document-specific queries
            sess = document_sessions.get(collection_name) or document_sessions.get(session_id) or {}
            result = await conversation_graph.chat(
                user_query=message,
                document_text=document_text,
                rag_context=rag_context,
                session_id=session_id,
                conversation_history=conversation_history,
                clause_index=clause_index if 'clause_index' in locals() else {},
                retrieval_metadata={
                    **(hybrid_retrieval_meta if 'hybrid_retrieval_meta' in locals() else {}),
                    "collection_name": collection_name,
                },
                collection_name=collection_name,
                document_name=sess.get("document_name", ""),
            )
        
        latency_metrics['conversation_processing'] = time.time() - step_start
        
        # Store assistant response in MongoDB
        step_start = time.time()
        if mongodb_service:
            try:
                await mongodb_service.add_chat_message(
                    session_id=session_id,
                    role="assistant",
                    content=result.get("response", ""),
                    intent=result.get("intent"),
                    agent_data=result.get("data"),
                    rag_used=rag_used,
                    rag_context=rag_context[:1000] if rag_context else None
                )
            except Exception as e:
                logger.warning(f"Could not store assistant message in MongoDB: {e}")
        latency_metrics['mongodb_write_assistant'] = time.time() - step_start
        
        # Calculate total latency
        total_latency = time.time() - start_time
        latency_metrics['total'] = total_latency
        
        # Log latency summary
        logger.info(f"⏱️ LATENCY SUMMARY:")
        logger.info(f"   Total: {total_latency:.2f}s")
        for step, duration in latency_metrics.items():
            if step != 'total':
                percentage = (duration / total_latency) * 100
                logger.info(f"   - {step}: {duration:.2f}s ({percentage:.1f}%)")
        
        return {
            "status": "success",
            "session_id": session_id,
            "collection_name": collection_name,
            "query": message,
            "intent": result.get("intent", "unknown"),
            "response": result.get("response", ""),
            "data": result.get("data", {}),
            "rag_info": {
                "used": rag_used,
                "collection": collection_name,
                "decision": rag_decision_reason,
                **(hybrid_retrieval_meta if rag_used and hybrid_mode else {}),
            },
            "latency_metrics": latency_metrics
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get information about a chat session from memory or MongoDB."""
    session = None
    source = None
    
    # Try in-memory first
    if session_id in document_sessions:
        session = document_sessions[session_id]
        source = "memory"
    
    # Try MongoDB if not in memory
    if session is None and mongodb_service:
        try:
            mongo_instance = await mongodb_service.get_instance(session_id)
            if mongo_instance:
                meta = mongo_instance.get("metadata") or {}
                session = {
                    "document_text": mongo_instance.get("document_text", ""),
                    "document_name": mongo_instance.get("document_name", "unknown"),
                    "template_text": mongo_instance.get("template_text"),
                    "template_name": mongo_instance.get("template_name"),
                    "collection_name": meta.get("collection_name"),
                }
                source = "mongodb"
                # Restore to in-memory for faster access
                document_sessions[session_id] = session
        except Exception as e:
            logger.warning(f"Could not retrieve session from MongoDB: {e}")
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get chat history stats from MongoDB
    chat_stats = {}
    if mongodb_service:
        try:
            stats = await mongodb_service.get_session_stats(session_id)
            chat_stats = {
                "message_count": stats.get("message_count", 0),
                "created_at": stats.get("created_at"),
                "updated_at": stats.get("updated_at")
            }
        except Exception as e:
            logger.warning(f"Could not get chat stats: {e}")
    
    return {
        "status": "success",
        "session_id": session_id,
        "document_name": session.get("document_name", "unknown"),
        "collection_name": session.get("collection_name"),
        "template_name": session.get("template_name"),
        "has_template": session.get("template_text") is not None,
        "document_length": len(session.get("document_text", "")),
        "source": source,
        "chat_stats": chat_stats
    }


@app.get("/session/{session_id}/history")
async def get_chat_history(
    session_id: str,
    limit: int = 50,
    include_data: bool = False
):
    """Get chat history for a session from MongoDB."""
    if not mongodb_service:
        raise HTTPException(status_code=503, detail="MongoDB service not available")
    
    try:
        history = await mongodb_service.get_chat_history(
            session_id=session_id,
            limit=limit,
            include_agent_data=include_data
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "message_count": len(history),
            "messages": history
        }
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chat history: {str(e)}")


@app.get("/sessions")
async def list_sessions(
    limit: int = 50,
    skip: int = 0
):
    """List all sessions from MongoDB."""
    if not mongodb_service:
        # Fall back to in-memory sessions
        sessions = [
            {
                "session_id": sid,
                "document_name": data.get("document_name", "unknown"),
                "collection_name": data.get("collection_name"),
                "template_name": data.get("template_name"),
                "has_template": data.get("template_text") is not None
            }
            for sid, data in list(document_sessions.items())[skip:skip+limit]
        ]
        return {
            "status": "success",
            "source": "memory",
            "count": len(sessions),
            "sessions": sessions
        }
    
    try:
        instances = await mongodb_service.list_instances(limit=limit, skip=skip)
        
        return {
            "status": "success",
            "source": "mongodb",
            "count": len(instances),
            "sessions": [
                {
                    "session_id": inst.get("session_id"),
                    "document_name": inst.get("document_name"),
                    "collection_name": (inst.get("metadata") or {}).get("collection_name"),
                    "template_name": inst.get("template_name"),
                    "has_template": inst.get("template_text") is not None,
                    "created_at": inst.get("created_at"),
                    "updated_at": inst.get("updated_at"),
                    "status": inst.get("status")
                }
                for inst in instances
            ]
        }
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
