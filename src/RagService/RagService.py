import os
import io
import base64
import fitz
import pandas as pd
import requests
import httpx
from PIL import Image
from bs4 import BeautifulSoup
from typing import List, Optional, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from qdrant_client.http import models as rest
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from embedding_service import OpenAIEmbeddings
from utils.logger import get_logger
from RagService.chunking import HierarchicalChunker
from RagService.retrieval import HierarchicalRetriever
from RagService.clause_aware_retrieval import ClauseAwareRetriever

load_dotenv()

logger = get_logger("RagService")

# Backward compatibility alias
USFEmbeddings = OpenAIEmbeddings


class RAGService:
    """
    RAG Service for chatbot with data ingestion and retrieval capabilities.
    Uses OpenAI embeddings and Qdrant vector database. USF is used for OCR only.
    """

    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None, openai_api_key: str = None):
        """
        Initialize RAG Service with Qdrant and OpenAI credentials.

        Args:
            qdrant_url: URL for Qdrant instance (defaults to env var)
            qdrant_api_key: API key for Qdrant (defaults to env var)
            openai_api_key: API key for OpenAI (defaults to env var)
        """
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.usf_api_key = os.getenv("USF_API_KEY")  # OCR/vision only

        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key or config.OPENAI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Initialize hierarchical chunking and retrieval
        from llm_service import LLMService
        self.llm_service = LLMService()
        self.chunking_llm_service = LLMService(model=config.CHUNKING_LLM_MODEL)
        self.hierarchical_chunker = HierarchicalChunker(llm_service=self.chunking_llm_service)
        # Note: HierarchicalRetriever will be created per-search with dynamic collection name
    
    def _image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    
    def _pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List[Image.Image]:
        """Convert PDF pages to PIL Images using PyMuPDF."""
        images = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            doc.close()
            logger.info(f"Converted {len(images)} pages from PDF")
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise Exception(f"Error converting PDF to images: {str(e)}")
        return images
    
    def _extract_text_from_image_sync(self, image: Image.Image) -> str:
        """Extract text from image using USF Vision API for OCR (sync)."""
        base64_image = self._image_to_base64(image)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.usf_api_key}"
        }
        
        api_url = config.USF_API_URL

        payload = {
            "model": config.OCR_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract ALL text from this image. Provide the complete text content exactly as it appears, preserving the structure and formatting."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "stream": False,
            "max_tokens": 4000
        }
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"USF API error: {response.text}")
        
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def data_ingestion_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF files using Image OCR with USF Vision API.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        try:
            logger.info(f"Starting OCR extraction from PDF: {pdf_path}")
            images = self._pdf_to_images(pdf_path)
            
            all_text = []
            for i, img in enumerate(images):
                logger.info(f"Processing page {i + 1}/{len(images)}")
                try:
                    page_text = self._extract_text_from_image_sync(img)
                    all_text.append(f"--- Page {i + 1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i + 1}: {e}")
                    all_text.append(f"--- Page {i + 1} ---\n[Error extracting text]")
            
            full_text = "\n\n".join(all_text)
            logger.info(f"OCR extraction complete. Total characters: {len(full_text)}")
            return full_text
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def data_ingestion_websites(self, url: str) -> str:
        """
        Extract text from websites using BeautifulSoup.
        
        Args:
            url: URL of the website
            
        Returns:
            Extracted text from the website
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            raise Exception(f"Error fetching website: {str(e)}")
    
    def data_ingestion_excel(self, excel_path: str) -> str:
        """
        Extract text from Excel files using pandas.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Extracted text from the Excel file
        """
        try:
            df = pd.read_excel(excel_path)
            text = df.to_string(index=False)
            return text
        except Exception as e:
            raise Exception(f"Error reading Excel: {str(e)}")
    
    def create_collection(self, collection_name: str):
        """
        Create a collection in Qdrant with OpenAI embedding dimensions and cosine metric.
        Also creates payload indexes for efficient filtering.
        
        Args:
            collection_name: Name of the collection to create
        """
        try:
            # Check if collection already exists
            collections = self.qdrant_client.get_collections().collections
            if any(col.name == collection_name for col in collections):
                print(f"Collection {collection_name} already exists.")
                # Ensure the indexes exist even if collection exists
                self._ensure_payload_indexes(collection_name)
                return
            
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=config.EMBED_DIMENSIONS, distance=Distance.COSINE)
            )
            print(f"Collection {collection_name} created successfully.")
            
            # Create payload indexes
            self._ensure_payload_indexes(collection_name)
            
        except Exception as e:
            raise Exception(f"Error creating collection: {str(e)}")
    
    def _ensure_payload_indexes(self, collection_name: str):
        """
        Ensure that required payload indexes exist for efficient filtering.
        Creates indexes for: source_collection, document_id, layer
        
        Args:
            collection_name: Name of the collection
        """
        indexes = [
            ("source_collection", PayloadSchemaType.KEYWORD),
            ("document_id", PayloadSchemaType.KEYWORD),
            ("chunk_id", PayloadSchemaType.KEYWORD),
            ("layer", PayloadSchemaType.INTEGER)
        ]
        
        for field_name, field_schema in indexes:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
                print(f"Created payload index on '{field_name}' for collection {collection_name}")
            except Exception as e:
                # Index might already exist, which is fine
                if "already exists" in str(e).lower():
                    print(f"Payload index on '{field_name}' already exists for collection {collection_name}")
                else:
                    print(f"Warning: Could not create payload index on '{field_name}': {str(e)}")
    
    def _ensure_source_collection_index(self, collection_name: str):
        """
        Legacy method - now calls _ensure_payload_indexes.
        Kept for backward compatibility.
        """
        self._ensure_payload_indexes(collection_name)
    
    def delete_collection(self, collection_name: str):
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name: Name of the collection to delete
        """
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            print(f"Collection {collection_name} deleted successfully.")
        except Exception as e:
            raise Exception(f"Error deleting collection: {str(e)}")
    
    def ensure_payload_indexes(self, collection_name: str = "main_collection"):
        """
        Ensure all required payload indexes exist on the collection.
        This is useful for fixing existing collections that don't have the indexes.
        
        Args:
            collection_name: Name of the collection (default: "main_collection")
        
        Returns:
            Dictionary with status information
        """
        try:
            # Ensure source_collection index exists
            self._ensure_source_collection_index(collection_name)
            
            return {
                "status": "success",
                "message": f"Payload indexes ensured for collection '{collection_name}'",
                "indexes_created": ["source_collection"]
            }
        except Exception as e:
            raise Exception(f"Error ensuring payload indexes: {str(e)}")
    
    def load_data_to_qdrant(
        self,
        collection_name: str,
        url_link: Optional[str] = None,
        pdf_file: Optional[str] = None,
        excel_file: Optional[str] = None
    ):
        """
        Load data into Qdrant using OpenAI Embeddings and recursive text splitter.
        
        Args:
            collection_name: Name of the collection
            url_link: URL to scrape (optional)
            pdf_file: Path to PDF file (optional)
            excel_file: Path to Excel file (optional)
        """
        try:
            # Ensure collection exists
            self.create_collection(collection_name)
            
            # Extract text based on source type
            text = ""
            if url_link:
                text = self.data_ingestion_websites(url_link)
            elif pdf_file:
                text = self.data_ingestion_pdf(pdf_file)
            elif excel_file:
                text = self.data_ingestion_excel(excel_file)
            else:
                raise ValueError("At least one data source must be provided")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Generate embeddings and prepare points
            points = []
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk)
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"text": chunk, "chunk_index": i}
                )
                points.append(point)
            
            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            print(f"Successfully loaded {len(chunks)} chunks to collection {collection_name}")
            return {"status": "success", "chunks_loaded": len(chunks)}
        
        except Exception as e:
            raise Exception(f"Error loading data to Qdrant: {str(e)}")
    
    def ingest_text(
        self,
        text: str,
        collection_name: str,
        metadata: Optional[dict] = None
    ):
        """
        Ingest raw text directly into Qdrant collection.
        
        Args:
            text: Raw text to ingest
            collection_name: Name of the collection to ingest into
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            dict with status and number of chunks loaded
        """
        try:
            # Ensure collection exists
            self.create_collection(collection_name)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                logger.warning(f"No chunks generated from text of length {len(text)}")
                return {"status": "success", "chunks_loaded": 0}
            
            # Generate embeddings and prepare points
            points = []
            base_metadata = metadata or {}
            
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk)
                payload = {
                    "text": chunk,
                    "chunk_index": i,
                    "source_collection": collection_name,
                    **base_metadata
                }
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Ingested {len(chunks)} chunks into collection '{collection_name}'")
            return {"status": "success", "chunks_loaded": len(chunks)}
        
        except Exception as e:
            logger.error(f"Error ingesting text to Qdrant: {str(e)}")
            raise Exception(f"Error ingesting text to Qdrant: {str(e)}")

    def retrieval_based_search(self, query: str, collections: Optional[List[str]] = None, top_k: int = 5):
        """
        Search for relevant documents across specified collections.
        
        Args:
            query: Search query
            collections: List of collection names to search in. 
                        If None or empty, searches in 'main_collection'.
            top_k: Number of top results to return
            
        Returns:
            List of search results with text, score, collection, and chunk_index
        """
        try:
            existing_collections = self.qdrant_client.get_collections().collections
            existing_names = [col.name for col in existing_collections]
            
            # Determine which collections to search
            if collections and len(collections) > 0:
                search_collections = [c for c in collections if c in existing_names]
            else:
                search_collections = ["main_collection"] if "main_collection" in existing_names else []
            
            if not search_collections:
                logger.warning(f"No valid collections found to search. Requested: {collections}, Available: {existing_names}")
                return []
            
            query_embedding = self.embeddings.embed_query(query)
            
            all_results = []
            for collection_name in search_collections:
                try:
                    search_results = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=top_k
                    )
                    
                    for result in search_results:
                        all_results.append({
                            "text": result.payload.get("text", ""),
                            "score": result.score,
                            "collection": collection_name,
                            "chunk_index": result.payload.get("chunk_index", 0)
                        })
                except Exception as e:
                    logger.warning(f"Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort by score and return top_k
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]

        except Exception as e:
            raise Exception(f"Error performing search: {str(e)}")

    
    async def async_data_ingestion_pdf(self, pdf_path: str) -> str:
        """
        Async extract text from PDF files using pdfplumber.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.data_ingestion_pdf, pdf_path)
    
    async def async_data_ingestion_websites(self, url: str) -> str:
        """
        Async extract text from websites using BeautifulSoup.
        
        Args:
            url: URL of the website
            
        Returns:
            Extracted text from the website
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.data_ingestion_websites, url)
    
    async def async_data_ingestion_excel(self, excel_path: str) -> str:
        """
        Async extract text from Excel files using pandas.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Extracted text from the Excel file
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.data_ingestion_excel, excel_path)
 

    async def load_data_to_qdrant_async(
        self,
        logical_collection_name: str,       # rename: this is NOT Qdrant collection
        url_links: Optional[List[str]] = None,
        pdf_files: Optional[List[str]] = None,
        excel_files: Optional[List[str]] = None
    ):
        """
        Load data into ONE Qdrant collection but keep logical separation using metadata.

        Args:
            logical_collection_name: User-defined logical group (e.g., "finance_docs")
            url_links: List of URLs to scrape (optional)
            pdf_files: List of PDF file paths (optional)
            excel_files: List of Excel file paths (optional)

        Returns:
            Dictionary with ingestion results
        """
        try:
            # Always use single Qdrant collection
            qdrant_collection = "main_collection"
            self.create_collection(qdrant_collection)

            tasks = []
            source_types = []

            if url_links:
                for url in url_links:
                    tasks.append(self.async_data_ingestion_websites(url))
                    source_types.append(f"URL: {url}")

            if pdf_files:
                for pdf in pdf_files:
                    tasks.append(self.async_data_ingestion_pdf(pdf))
                    source_types.append(f"PDF: {pdf}")

            if excel_files:
                for excel in excel_files:
                    tasks.append(self.async_data_ingestion_excel(excel))
                    source_types.append(f"Excel: {excel}")

            if not tasks:
                raise ValueError("At least one data source must be provided")

            print(f"Starting parallel ingestion of {len(tasks)} sources...")
            texts = await asyncio.gather(*tasks, return_exceptions=True)

            all_chunks = []
            successful_sources = []
            failed_sources = []

            for text, source_type in zip(texts, source_types):
                if isinstance(text, Exception):
                    print(f"Failed to ingest {source_type}: {str(text)}")
                    failed_sources.append({"source": source_type, "error": str(text)})
                    continue

                chunks = self.text_splitter.split_text(text)
                all_chunks.extend(chunks)
                successful_sources.append({"source": source_type, "chunks": len(chunks)})
                print(f"Extracted {len(chunks)} chunks from {source_type}")

            if not all_chunks:
                raise Exception("No data extracted from any source")

            print(f"Generating embeddings for {len(all_chunks)} chunks...")

            points = []
            for i, chunk in enumerate(all_chunks):
                embedding = self.embeddings.embed_query(chunk)
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "chunk_index": i,
                            "source_collection": logical_collection_name     # << ADD THIS
                        }
                    )
                )

            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=qdrant_collection,
                points=points
            )

            print(f"Uploaded {len(points)} chunks into Qdrant under group '{logical_collection_name}'")

            return {
                "status": "success",
                "total_chunks_loaded": len(points),
                "sources_processed": len(successful_sources),
                "sources_failed": len(failed_sources),
                "successful_sources": successful_sources,
                "failed_sources": failed_sources if failed_sources else None
            }

        except Exception as e:
            raise Exception(f"Error loading data to Qdrant: {str(e)}")
    
    async def ingest_hierarchical_chunks(
        self,
        text: str = None,
        collection_name: str = "legal_documents",
        document_id: Optional[str] = None,
        chunking_result: dict = None
    ):
        """
        Process document using hierarchical chunking and ingest all layers into Qdrant.
        
        Args:
            text: Document text to process (ignored if chunking_result provided)
            collection_name: Qdrant collection name
            document_id: Optional document identifier
            chunking_result: Pre-computed chunking result (skips re-chunking if provided)
        
        Returns:
            Dictionary with ingestion results and tree structure
        """
        try:
            # Ensure collection exists
            self.create_collection(collection_name)
            
            # Use pre-computed chunks if provided, otherwise run chunking
            if chunking_result is not None:
                logger.info(f"Using pre-computed chunking result (skipping re-chunking)")
                result = chunking_result
            else:
                if text is None:
                    raise ValueError("Either 'text' or 'chunking_result' must be provided")
                logger.info(f"Starting hierarchical chunking for document (length: {len(text)} chars)")
                result = await self.hierarchical_chunker.process_document(text)
            
            # Prepare points for all chunks across all layers
            points = []
            all_chunks = []
            
            for layer_name, chunks in result['layers'].items():
                all_chunks.extend(chunks)
            
            logger.info(f"Processing {len(all_chunks)} chunks across {result['num_layers']} layers")
            
            # Filter out empty chunks first
            valid_chunks = []
            skipped_chunks = 0
            for chunk in all_chunks:
                if not chunk.text or not chunk.text.strip():
                    logger.warning(f"Skipping chunk {chunk.chunk_id} - empty text field")
                    skipped_chunks += 1
                else:
                    valid_chunks.append(chunk)
            
            if not valid_chunks:
                logger.warning("No valid chunks to ingest")
                return {"status": "success", "total_chunks": 0, "num_layers": result['num_layers']}
            
            # OPTIMIZATION: Generate all embeddings in parallel
            logger.info(f"Generating embeddings for {len(valid_chunks)} chunks in parallel...")
            chunk_texts = [chunk.text for chunk in valid_chunks]
            
            try:
                # Use async parallel embedding
                embeddings = await self.embeddings.async_embed_documents(chunk_texts)
                logger.info(f"Generated {len(embeddings)} embeddings")
            except Exception as e:
                logger.error(f"Parallel embedding failed: {e}")
                raise
            
            # Create points with embeddings
            for chunk, embedding in zip(valid_chunks, embeddings):
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "layer": chunk.layer,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "clauses": chunk.clauses,
                    "clause_relationships": chunk.clause_relationships,
                    "risk_level": chunk.risk_level,
                    "parent_id": chunk.parent_id,
                    "child_ids": chunk.child_ids,
                    "short_summary": chunk.short_summary,
                    "metadata": chunk.metadata,
                }
                if chunk.clause_index:
                    payload["clause_index"] = chunk.clause_index
                meta = chunk.metadata or {}
                if meta.get("clause_type_counts"):
                    payload["clause_type_counts"] = meta["clause_type_counts"]
                
                if document_id:
                    payload["document_id"] = document_id
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            if skipped_chunks > 0:
                logger.warning(f"Skipped {skipped_chunks} chunks due to empty text")
            
            # Upload to Qdrant in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            logger.info(f"Successfully ingested {len(points)} chunks into '{collection_name}'")
            
            tree = result.get("tree_structure") or {}
            global_index = tree.get("global_clause_index") or {}
            clause_type_counts = tree.get("clause_type_counts") or {}
            if not clause_type_counts and global_index:
                for _cid, info in global_index.items():
                    ct = info.get("clause_type", "Unknown")
                    clause_type_counts[ct] = clause_type_counts.get(ct, 0) + 1

            return {
                "status": "success",
                "total_chunks": len(points),
                "num_layers": result['num_layers'],
                "tree_structure": tree,
                "global_clause_index": global_index,
                "clause_type_counts": clause_type_counts,
                "clause_index_entries": len(global_index),
                "layer_distribution": {
                    layer_name: len(chunks)
                    for layer_name, chunks in result['layers'].items()
                },
            }
        
        except Exception as e:
            logger.error(f"Error in hierarchical ingestion: {str(e)}")
            raise Exception(f"Error in hierarchical ingestion: {str(e)}")
    
    async def hierarchical_search(
        self,
        query: str,
        collection_name: str = "legal_documents",
        top_k: int = 3,
        include_context: bool = True
    ):
        """
        Search using hierarchical retrieval.
        Searches L2+ layers (excludes only L1 raw chunks) for better coverage.
        
        Args:
            query: Search query text
            collection_name: Qdrant collection name
            top_k: Number of results to return
            include_context: Whether to include parent/child context
        
        Returns:
            List of search results from L2+ layers
        """
        try:
            logger.info(f"Hierarchical search: '{query[:100]}...'")
            
            # OPTIMIZATION: Use async embedding for faster query processing
            query_vector = await self.embeddings.async_embed_query(query)
            
            # Create retriever for the specific collection
            retriever = HierarchicalRetriever(
                collection_name=collection_name,
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                top_k=top_k
            )
            
            # Search using hierarchical retriever
            if include_context:
                results = retriever.search_with_context(
                    query_vector=query_vector,
                    include_parents=True,
                    include_children=False,
                    top_k=top_k
                )
            else:
                results = retriever.search(
                    query_vector=query_vector,
                    top_k=top_k
                )
            
            logger.info(f"Found {len(results)} results from L2+ layers")
            return results
        
        except Exception as e:
            logger.error(f"Error in hierarchical search: {str(e)}")
            raise Exception(f"Error in hierarchical search: {str(e)}")
    
    async def hybrid_search(
        self,
        query: str,
        collection_name: str = "legal_documents",
        top_k: int = 5,
        include_context: bool = True
    ):
        """
        Hybrid search combining semantic search + clause-aware retrieval.
        
        This performs two searches:
        1. Semantic search (standard vector similarity)
        2. Clause-aware search (with intent-based filtering)
        
        Results are merged and deduplicated, with clause-aware matches boosted.
        
        Args:
            query: Search query text
            collection_name: Qdrant collection name
            top_k: Number of results to return per search type
            include_context: Whether to include parent/child context
        
        Returns:
            Merged and ranked results from both search strategies
        """
        try:
            logger.info(f"🔀 Hybrid search: '{query[:100]}...'")
            
            # Generate query embedding (async for speed)
            query_vector = await self.embeddings.async_embed_query(query)
            
            # 1. SEMANTIC SEARCH (standard hierarchical retrieval)
            logger.info("📊 Running semantic search...")
            semantic_retriever = HierarchicalRetriever(
                collection_name=collection_name,
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                top_k=top_k
            )
            
            if include_context:
                semantic_results = semantic_retriever.search_with_context(
                    query_vector=query_vector,
                    include_parents=True,
                    include_children=False,
                    top_k=top_k
                )
            else:
                semantic_results = semantic_retriever.search(
                    query_vector=query_vector,
                    top_k=top_k
                )
            
            logger.info(f"   Semantic search: {len(semantic_results)} results")
            
            # 2. CLAUSE-AWARE SEARCH (with intent-based filtering)
            logger.info("🎯 Running clause-aware search...")
            clause_retriever = ClauseAwareRetriever(
                collection_name=collection_name,
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key
            )
            
            # Analyze query intent
            intent = clause_retriever.analyze_query_intent(query)
            logger.info(f"   Query intent: clause_types={intent['detected_clause_types']}, "
                       f"risk_levels={intent['detected_risk_levels']}")
            
            clause_results = clause_retriever.search_with_clause_awareness(
                query_vector=query_vector,
                query_text=query,
                top_k=top_k
            )
            
            logger.info(f"   Clause-aware search: {len(clause_results)} results")
            
            # 3. MERGE AND DEDUPLICATE RESULTS
            merged_results = self._merge_hybrid_results(
                semantic_results=semantic_results,
                clause_results=clause_results,
                intent=intent,
                top_k=top_k
            )
            
            logger.info(f"✅ Hybrid search complete: {len(merged_results)} merged results")
            return merged_results
        
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise Exception(f"Error in hybrid search: {str(e)}")
    
    def _merge_hybrid_results(
        self,
        semantic_results: List[dict],
        clause_results: List[dict],
        intent: dict,
        top_k: int = 5
    ) -> List[dict]:
        """
        Merge and rank results from semantic and clause-aware searches.
        
        Scoring:
        - Base score from vector similarity
        - Boost for clause-aware matches
        - Boost for matching clause types
        - Boost for matching risk levels
        
        Args:
            semantic_results: Results from semantic search
            clause_results: Results from clause-aware search
            intent: Query intent analysis
            top_k: Number of final results to return
            
        Returns:
            Merged and ranked results
        """
        seen_chunks = {}
        
        # Process semantic results
        for result in semantic_results:
            chunk_id = result.get("chunk_id")
            if chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = {
                    **result,
                    "hybrid_score": result.get("score", 0),
                    "search_sources": ["semantic"],
                    "intent_match": intent
                }
            else:
                seen_chunks[chunk_id]["search_sources"].append("semantic")
        
        # Process clause-aware results with boost
        CLAUSE_BOOST = 0.15  # Boost for clause-aware match
        MATCHING_CLAUSE_BOOST = 0.1  # Additional boost for matching clause types
        
        for result in clause_results:
            chunk_id = result.get("chunk_id")
            base_score = result.get("score", 0)
            
            # Calculate boost
            boost = CLAUSE_BOOST
            
            # Additional boost if matching clauses found
            if result.get("matching_clauses"):
                boost += MATCHING_CLAUSE_BOOST * min(len(result["matching_clauses"]), 3)
            
            boosted_score = base_score + boost
            
            if chunk_id not in seen_chunks:
                seen_chunks[chunk_id] = {
                    **result,
                    "hybrid_score": boosted_score,
                    "search_sources": ["clause_aware"],
                    "intent_match": intent
                }
            else:
                # Update score if clause-aware score is higher
                if boosted_score > seen_chunks[chunk_id]["hybrid_score"]:
                    seen_chunks[chunk_id]["hybrid_score"] = boosted_score
                seen_chunks[chunk_id]["search_sources"].append("clause_aware")
                
                # Add matching clauses info
                if result.get("matching_clauses"):
                    seen_chunks[chunk_id]["matching_clauses"] = result["matching_clauses"]
        
        # Sort by hybrid score and return top_k
        sorted_results = sorted(
            seen_chunks.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        
        logger.info(f"Merged {len(semantic_results)} semantic + {len(clause_results)} clause-aware "
                   f"-> {len(sorted_results)} unique results")
        
        return sorted_results[:top_k]

