"""
Retrieval service for hierarchical chunking using Qdrant vector database.
Filters out L1 and L2 layers, retrieves from L3+ layers only.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from utils.logger import get_logger

logger = get_logger("RetrievalService")


class HierarchicalRetriever:
    """
    Retrieval service for hierarchical chunking system.
    
    Filters chunks to exclude L1 (original chunks) and L2 (1:1 summaries).
    Only retrieves from L3+ layers (grouped summaries with context).
    """
    
    def __init__(
        self,
        collection_name: str = "legal_documents",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        top_k: int = 3
    ):
        """
        Initialize the hierarchical retriever.
        
        Args:
            collection_name: Name of the Qdrant collection
            qdrant_url: Qdrant server URL (defaults to config)
            qdrant_api_key: Qdrant API key (defaults to config)
            top_k: Number of top results to retrieve (default: 3)
        """
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Initialize Qdrant client
        self.qdrant_url = qdrant_url or config.QDRANT_URL
        self.qdrant_api_key = qdrant_api_key or config.QDRANT_API_KEY
        
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL not configured. Please set it in .env file.")
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key if self.qdrant_api_key else None
        )
        
        logger.info(f"HierarchicalRetriever initialized with collection: {collection_name}, top_k: {top_k}")
    
    def _create_layer_filter(self) -> Filter:
        """
        Create filter to exclude L1 layer only.
        Retrieves from L2+ (summaries with clauses and higher layers).
        
        Returns:
            Qdrant Filter object
        """
        # Filter: layer >= 2 (exclude only L1 raw chunks)
        # This gives us more results while still filtering out raw text
        layer_filter = Filter(
            must=[
                FieldCondition(
                    key="layer",
                    range=Range(gte=2)  # Greater than or equal to 2
                )
            ]
        )
        
        logger.debug("Created layer filter: layer >= 2 (excluding L1 raw chunks)")
        return layer_filter
    
    def search(
        self,
        query_vector: List[float],
        additional_filters: Optional[Filter] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        Automatically filters out L1 and L2 layers.
        
        Args:
            query_vector: Query embedding vector
            additional_filters: Optional additional filters to apply
            top_k: Number of results to return (overrides default)
        
        Returns:
            List of search results with metadata
        """
        k = top_k if top_k is not None else self.top_k
        
        # Create base layer filter (exclude L1 and L2)
        layer_filter = self._create_layer_filter()
        
        # Combine with additional filters if provided
        if additional_filters:
            # Merge filters
            combined_filter = Filter(
                must=layer_filter.must + (additional_filters.must or []),
                should=additional_filters.should,
                must_not=additional_filters.must_not
            )
        else:
            combined_filter = layer_filter
        
        logger.info(f"Searching collection '{self.collection_name}' with top_k={k}")
        logger.debug(f"Applied filters: {combined_filter}")
        
        try:
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=combined_filter,
                limit=k,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = []
            for hit in search_results:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "chunk_id": hit.payload.get("chunk_id"),
                    "layer": hit.payload.get("layer"),
                    "text": hit.payload.get("text"),
                    "clauses": hit.payload.get("clauses", []),
                    "clause_relationships": hit.payload.get("clause_relationships", []),
                    "risk_level": hit.payload.get("risk_level"),
                    "parent_id": hit.payload.get("parent_id"),
                    "child_ids": hit.payload.get("child_ids", []),
                    "short_summary": hit.payload.get("short_summary"),
                    "metadata": hit.payload.get("metadata", {})
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} results from layers L2+")
            
            # Log layer distribution
            layer_counts = {}
            for r in results:
                layer = r.get("layer", "unknown")
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            logger.debug(f"Layer distribution: {layer_counts}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def search_by_text(
        self,
        query_text: str,
        embedding_function,
        additional_filters: Optional[Filter] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (converts to vector using embedding function).
        
        Args:
            query_text: Text query to search for
            embedding_function: Function to convert text to vector
            additional_filters: Optional additional filters
            top_k: Number of results to return
        
        Returns:
            List of search results
        """
        logger.info(f"Converting query text to vector: '{query_text[:100]}...'")
        
        # Convert text to vector
        query_vector = embedding_function(query_text)
        
        # Perform search
        return self.search(
            query_vector=query_vector,
            additional_filters=additional_filters,
            top_k=top_k
        )
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by its chunk_id.
        
        Args:
            chunk_id: The chunk_id to retrieve
        
        Returns:
            Chunk data or None if not found
        """
        logger.info(f"Retrieving chunk by ID: {chunk_id}")
        
        try:
            # Search with exact match filter
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchValue(value=chunk_id)
                    )
                ]
            )
            
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if results[0]:  # results is a tuple (points, next_offset)
                point = results[0][0]
                return {
                    "id": point.id,
                    "chunk_id": point.payload.get("chunk_id"),
                    "layer": point.payload.get("layer"),
                    "text": point.payload.get("text"),
                    "clauses": point.payload.get("clauses", []),
                    "clause_relationships": point.payload.get("clause_relationships", []),
                    "risk_level": point.payload.get("risk_level"),
                    "parent_id": point.payload.get("parent_id"),
                    "child_ids": point.payload.get("child_ids", []),
                    "short_summary": point.payload.get("short_summary"),
                    "metadata": point.payload.get("metadata", {})
                }
            
            logger.warning(f"Chunk not found: {chunk_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            raise
    
    def get_parent_chunks(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        Get parent chunks by following parent_id references.
        
        Args:
            chunk_id: Starting chunk ID
        
        Returns:
            List of parent chunks (from immediate parent up to L1)
        """
        parents = []
        current_chunk = self.get_chunk_by_id(chunk_id)
        
        while current_chunk and current_chunk.get("parent_id"):
            parent_id = current_chunk["parent_id"]
            parent_chunk = self.get_chunk_by_id(parent_id)
            
            if parent_chunk:
                parents.append(parent_chunk)
                current_chunk = parent_chunk
            else:
                break
        
        logger.info(f"Found {len(parents)} parent chunks for {chunk_id}")
        return parents
    
    def get_child_chunks(self, chunk_id: str) -> List[Dict[str, Any]]:
        """
        Get child chunks by following child_ids references.
        
        Args:
            chunk_id: Parent chunk ID
        
        Returns:
            List of child chunks
        """
        chunk = self.get_chunk_by_id(chunk_id)
        
        if not chunk:
            return []
        
        child_ids = chunk.get("child_ids", [])
        children = []
        
        for child_id in child_ids:
            child_chunk = self.get_chunk_by_id(child_id)
            if child_chunk:
                children.append(child_chunk)
        
        logger.info(f"Found {len(children)} child chunks for {chunk_id}")
        return children
    
    def search_with_context(
        self,
        query_vector: List[float],
        include_parents: bool = True,
        include_children: bool = False,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search and optionally include parent/child chunks for context.
        
        Args:
            query_vector: Query embedding vector
            include_parents: Include parent chunks in results
            include_children: Include child chunks in results
            top_k: Number of initial results
        
        Returns:
            List of results with context chunks
        """
        # Get initial search results
        results = self.search(query_vector=query_vector, top_k=top_k)
        
        if not (include_parents or include_children):
            return results
        
        # Enrich with context
        enriched_results = []
        for result in results:
            enriched = result.copy()
            
            if include_parents:
                enriched["parent_chunks"] = self.get_parent_chunks(result["chunk_id"])
            
            if include_children:
                enriched["child_chunks"] = self.get_child_chunks(result["chunk_id"])
            
            enriched_results.append(enriched)
        
        logger.info(f"Enriched {len(enriched_results)} results with context")
        return enriched_results


# Example usage
if __name__ == "__main__":
    # Initialize retriever
    retriever = HierarchicalRetriever(
        collection_name="legal_documents",
        top_k=3
    )
    
    # Example: Search with a query vector
    # query_vector = [0.1, 0.2, 0.3, ...]  # Your embedding vector
    # results = retriever.search(query_vector)
    
    # Example: Get chunk by ID
    # chunk = retriever.get_chunk_by_id("L3_summary_0")
    
    # Example: Search with parent context
    # results = retriever.search_with_context(
    #     query_vector=query_vector,
    #     include_parents=True,
    #     top_k=3
    # )
    
    print("HierarchicalRetriever initialized successfully")
    print(f"Collection: {retriever.collection_name}")
    print(f"Top K: {retriever.top_k}")
    print("Filter: Excludes L1 and L2, retrieves from L3+ only")
