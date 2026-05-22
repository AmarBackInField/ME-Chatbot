"""
Root Clause-Aware Retrieval Strategy

This module implements a retrieval strategy that:
1. First finds the root chunk (final layer) in the collection
2. Extracts the clause_index from the root chunk (contains all clauses with source tracking)
3. Uses LLM to analyze user query and match to relevant clauses
4. Retrieves the source chunks for matched clauses

This approach ensures the agent always has visibility into ALL clauses
in the document before deciding which chunks to retrieve.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from utils.logger import get_logger

# Import LLMService - handle different import paths
try:
    from LLMService.LLMService import LLMService
except ImportError:
    try:
        from services.LLMService import LLMService
    except ImportError:
        from llm_service import LLMService

logger = get_logger("RootClauseRetrieval")


class RootClauseRetriever:
    """
    Retrieval strategy that uses the root chunk's clause_index to guide retrieval.
    
    Flow:
    1. Find root chunk (highest layer, single chunk)
    2. Extract clause_index (all clauses with source_chunk_id)
    3. LLM analyzes query + clause_index to select relevant clauses
    4. Retrieve source chunks for selected clauses
    """
    
    def __init__(
        self,
        collection_name: str,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or config.QDRANT_URL
        self.qdrant_api_key = qdrant_api_key or config.QDRANT_API_KEY
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key if self.qdrant_api_key else None
        )
        
        self.llm_service = llm_service or LLMService()
        
        # Cache for root chunk data
        self._root_chunk_cache = None
        self._clause_index_cache = None
        
        logger.info(f"RootClauseRetriever initialized for collection: {collection_name}")
    
    async def get_root_chunk(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Find and return the root chunk (final layer, single chunk).
        
        Args:
            force_refresh: Force refresh from database even if cached
            
        Returns:
            Root chunk payload or None if not found
        """
        if self._root_chunk_cache and not force_refresh:
            return self._root_chunk_cache
        
        try:
            # First, find the maximum layer in the collection
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            if not results[0]:
                logger.warning(f"No chunks found in collection: {self.collection_name}")
                return None
            
            # Find the highest layer
            max_layer = 0
            for point in results[0]:
                layer = point.payload.get("layer", 0)
                if layer > max_layer:
                    max_layer = layer
            
            logger.info(f"Found max layer: {max_layer}")
            
            # Get chunks from the highest layer
            root_chunks = [
                point for point in results[0]
                if point.payload.get("layer") == max_layer
            ]
            
            if not root_chunks:
                logger.warning(f"No root chunks found at layer {max_layer}")
                return None
            
            # Should be single root chunk, but take first if multiple
            root_chunk = root_chunks[0]
            self._root_chunk_cache = root_chunk.payload
            
            logger.info(f"Root chunk found: {root_chunk.payload.get('chunk_id')} "
                       f"(layer {max_layer}, {len(root_chunk.payload.get('clauses', []))} clauses)")
            
            return self._root_chunk_cache
            
        except Exception as e:
            logger.error(f"Error finding root chunk: {e}")
            return None
    
    async def get_clause_index(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get the clause_index from the root chunk.
        
        Args:
            force_refresh: Force refresh from database
            
        Returns:
            Dictionary mapping clause_id -> clause info
        """
        if self._clause_index_cache and not force_refresh:
            return self._clause_index_cache
        
        root_chunk = await self.get_root_chunk(force_refresh)
        if not root_chunk:
            return {}
        
        self._clause_index_cache = root_chunk.get("clause_index", {})
        
        # Also build from clauses array if clause_index is empty
        if not self._clause_index_cache:
            clauses = root_chunk.get("clauses", [])
            self._clause_index_cache = {}
            for clause in clauses:
                clause_id = clause.get("clause_id", "")
                if clause_id:
                    self._clause_index_cache[clause_id] = {
                        "source_chunk_id": clause.get("source_chunk_id", ""),
                        "clause_type": clause.get("clause_type", ""),
                        "parent_clause_id": clause.get("parent_clause_id"),
                        "importance": clause.get("importance", "medium"),
                        "content": clause.get("content", "")
                    }
        
        logger.info(f"Clause index loaded: {len(self._clause_index_cache)} clauses")
        return self._clause_index_cache
    
    async def analyze_query_with_llm(
        self,
        query: str,
        clause_index: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Use LLM to analyze query and select relevant clauses from the index.
        
        IMPROVED: Better prompting, retry logic, and keyword fallback.
        
        Args:
            query: User query
            clause_index: Dictionary of all available clauses
            
        Returns:
            List of selected clause_ids
        """
        import re
        
        # Format clause index for LLM - simplified format
        clause_list = []
        for clause_id, info in clause_index.items():
            clause_list.append(f"- {clause_id}: {info.get('clause_type', 'Unknown')} ({info.get('importance', 'medium')})")
        
        clauses_text = "\n".join(clause_list)
        
        prompt = f"""You are a legal document clause selector. Select ALL clauses relevant to the user's question.

USER QUESTION: {query}

AVAILABLE CLAUSES:
{clauses_text}

TASK: Return a JSON array of clause_ids that are relevant to answering this question.
- Be INCLUSIVE - select all potentially relevant clauses
- Include parent clauses when selecting sub-clauses
- Look for semantic matches (e.g., "termination" matches "Termination", "End of Employment", etc.)

IMPORTANT: Return ONLY a valid JSON array, nothing else.
Example: ["L2_C0_C1", "L2_C0_C2", "L2_C1_C3"]

Your response (JSON array only):"""

        max_retries = 2
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": "You are a JSON-only response bot. Return only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ]
                
                response = await self.llm_service.chat_completion(messages, max_tokens=500, temperature=0.1)
                
                if not response or not response.strip():
                    logger.warning(f"Empty LLM response on attempt {attempt + 1}")
                    continue
                
                # Clean response
                response_clean = response.strip()
                
                # Remove markdown code blocks
                if '```' in response_clean:
                    response_clean = re.sub(r'```json?\s*', '', response_clean)
                    response_clean = re.sub(r'\s*```', '', response_clean)
                    response_clean = response_clean.strip()
                
                # Extract JSON array if embedded in text
                array_match = re.search(r'\[.*?\]', response_clean, re.DOTALL)
                if array_match:
                    response_clean = array_match.group(0)
                
                selected_clauses = json.loads(response_clean)
                
                if not isinstance(selected_clauses, list):
                    logger.warning(f"LLM returned non-list: {type(selected_clauses)}")
                    continue
                
                # Validate clause IDs exist
                valid_clauses = [c for c in selected_clauses if c in clause_index]
                
                if valid_clauses:
                    logger.info(f"LLM selected {len(valid_clauses)} clauses for query: {query[:50]}...")
                    return valid_clauses
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
            except Exception as e:
                logger.error(f"Error in LLM clause selection attempt {attempt + 1}: {e}")
        
        # Fallback: keyword-based matching
        logger.info(f"Using keyword fallback for query: {query[:50]}...")
        return self._keyword_clause_match(query, clause_index)
    
    def _keyword_clause_match(
        self,
        query: str,
        clause_index: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Fallback keyword-based clause matching when LLM fails.
        """
        import re
        
        # Normalize query
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Keywords mapping to clause types
        keyword_mappings = {
            # Employment basics
            "job": ["position", "duties", "role", "employment", "appointment"],
            "title": ["position", "role", "designation"],
            "responsibilities": ["duties", "obligations", "responsibilities"],
            "salary": ["salary", "compensation", "remuneration", "pay", "wage"],
            "compensation": ["salary", "compensation", "remuneration", "benefits", "pay"],
            "payment": ["salary", "payment", "compensation"],
            
            # Time-related
            "hours": ["hours", "working", "schedule", "time"],
            "schedule": ["hours", "working", "schedule"],
            "working": ["hours", "working", "employment"],
            
            # Termination
            "termination": ["termination", "end", "resignation", "dismissal", "notice"],
            "notice": ["notice", "termination", "period"],
            "resignation": ["termination", "resignation", "end"],
            
            # Confidentiality
            "confidentiality": ["confidential", "non-disclosure", "nda", "secret", "proprietary"],
            "confidential": ["confidential", "non-disclosure", "secret"],
            "non-disclosure": ["confidential", "non-disclosure", "nda"],
            "nda": ["confidential", "non-disclosure", "nda"],
            
            # IP
            "intellectual": ["intellectual", "property", "ip", "invention", "patent"],
            "property": ["intellectual", "property", "ip"],
            "ip": ["intellectual", "property", "ip", "invention"],
            "invention": ["intellectual", "invention", "ip"],
            
            # Non-compete
            "non-compete": ["non-compete", "competition", "restrictive"],
            "compete": ["non-compete", "competition"],
            "non-solicitation": ["non-solicitation", "solicitation", "restrictive"],
            "solicitation": ["non-solicitation", "solicitation"],
            
            # Leave
            "leave": ["leave", "vacation", "holiday", "absence", "pto"],
            "vacation": ["leave", "vacation", "holiday", "pto"],
            "holiday": ["leave", "vacation", "holiday"],
            
            # Benefits
            "benefits": ["benefits", "insurance", "health", "medical"],
            "insurance": ["benefits", "insurance", "health", "medical"],
            "health": ["benefits", "insurance", "health", "medical"],
            
            # Probation
            "probation": ["probation", "trial", "probationary"],
            "probationary": ["probation", "probationary"],
            
            # Dispute
            "dispute": ["dispute", "arbitration", "resolution", "mediation"],
            "arbitration": ["dispute", "arbitration", "resolution"],
            "resolution": ["dispute", "resolution", "arbitration"],
            
            # Legal
            "governing": ["governing", "law", "jurisdiction", "applicable"],
            "jurisdiction": ["governing", "jurisdiction", "law"],
            "law": ["governing", "law", "jurisdiction"],
            
            # Amendment
            "amendment": ["amendment", "modification", "change", "variation"],
            "modification": ["amendment", "modification", "change"],
            
            # Severability
            "severability": ["severability", "invalid", "provisions"],
            "waiver": ["waiver", "rights", "provisions"],
        }
        
        # Find matching keywords
        search_terms = set()
        for word in query_words:
            if word in keyword_mappings:
                search_terms.update(keyword_mappings[word])
            else:
                search_terms.add(word)
        
        # Match clauses
        matched_clauses = []
        for clause_id, info in clause_index.items():
            clause_type = info.get("clause_type", "").lower()
            content = info.get("content", "").lower()
            
            # Check if any search term matches
            for term in search_terms:
                if term in clause_type or term in content:
                    matched_clauses.append(clause_id)
                    break
        
        # If no matches, return top 5 high-importance clauses
        if not matched_clauses:
            high_importance = [
                cid for cid, info in clause_index.items()
                if info.get("importance") == "high" and info.get("parent_clause_id") is None
            ]
            matched_clauses = high_importance[:5]
        
        logger.info(f"Keyword fallback matched {len(matched_clauses)} clauses")
        return matched_clauses
    
    async def get_chunks_for_clauses(
        self,
        clause_ids: List[str],
        clause_index: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve source chunks for the selected clauses.
        
        Args:
            clause_ids: List of selected clause IDs
            clause_index: Clause index for source lookup
            
        Returns:
            List of chunk payloads
        """
        # Get unique source chunk IDs
        source_chunk_ids = set()
        for clause_id in clause_ids:
            if clause_id in clause_index:
                source_id = clause_index[clause_id].get("source_chunk_id", "")
                if source_id:
                    source_chunk_ids.add(source_id)
        
        if not source_chunk_ids:
            logger.warning("No source chunks found for selected clauses")
            return []
        
        logger.info(f"Retrieving {len(source_chunk_ids)} source chunks: {source_chunk_ids}")
        
        # Retrieve chunks by chunk_id
        chunks = []
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            for point in results[0]:
                chunk_id = point.payload.get("chunk_id", "")
                if chunk_id in source_chunk_ids:
                    # Add matched clauses info
                    matched_clauses = [
                        clause_index[cid] for cid in clause_ids
                        if clause_index.get(cid, {}).get("source_chunk_id") == chunk_id
                    ]
                    
                    chunk_data = {
                        **point.payload,
                        "matched_clause_ids": [
                            cid for cid in clause_ids
                            if clause_index.get(cid, {}).get("source_chunk_id") == chunk_id
                        ],
                        "matched_clauses": matched_clauses
                    }
                    chunks.append(chunk_data)
            
            logger.info(f"Retrieved {len(chunks)} chunks for {len(clause_ids)} clauses")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_root_summary: bool = True
    ) -> Dict[str, Any]:
        """
        Main retrieval method using root clause-aware strategy.
        
        Args:
            query: User query
            top_k: Maximum number of chunks to return
            include_root_summary: Include root chunk summary in response
            
        Returns:
            Dictionary with retrieval results
        """
        logger.info(f"Starting root clause-aware retrieval for: {query[:50]}...")
        
        # Step 1: Get root chunk and clause index
        root_chunk = await self.get_root_chunk()
        if not root_chunk:
            return {
                "status": "error",
                "message": "No root chunk found in collection",
                "chunks": []
            }
        
        clause_index = await self.get_clause_index()
        if not clause_index:
            return {
                "status": "warning",
                "message": "No clause index found in root chunk",
                "chunks": [],
                "root_summary": root_chunk.get("text", "")[:500] if include_root_summary else None
            }
        
        # Step 2: LLM analyzes query and selects relevant clauses
        selected_clause_ids = await self.analyze_query_with_llm(query, clause_index)
        
        if not selected_clause_ids:
            # Fallback: return root chunk summary
            return {
                "status": "no_match",
                "message": "No specific clauses matched the query",
                "chunks": [],
                "root_summary": root_chunk.get("text", "")[:1000] if include_root_summary else None,
                "all_clauses": list(clause_index.keys())
            }
        
        # Step 3: Retrieve source chunks for selected clauses
        chunks = await self.get_chunks_for_clauses(selected_clause_ids, clause_index)
        
        # Limit to top_k
        chunks = chunks[:top_k]
        
        return {
            "status": "success",
            "query": query,
            "selected_clauses": selected_clause_ids,
            "clause_details": {
                cid: clause_index.get(cid, {}) for cid in selected_clause_ids
            },
            "chunks": chunks,
            "num_chunks": len(chunks),
            "root_summary": root_chunk.get("short_summary", "")[:500] if include_root_summary else None,
            "total_clauses_available": len(clause_index)
        }
    
    def get_retrieval_explanation(self) -> str:
        """
        Get a human-readable explanation of how this retrieval strategy works.
        """
        return """
ROOT CLAUSE-AWARE RETRIEVAL STRATEGY
=====================================

How it works:
1. FIND ROOT CHUNK
   - Locate the final layer chunk (document corpus summary)
   - This chunk contains ALL clauses aggregated from lower layers

2. EXTRACT CLAUSE INDEX
   - Get the clause_index from root chunk
   - Each clause has: clause_id, clause_type, source_chunk_id, parent_clause_id

3. LLM CLAUSE SELECTION
   - Present user query + all available clauses to LLM
   - LLM selects most relevant clauses for the query

4. RETRIEVE SOURCE CHUNKS
   - Use source_chunk_id to fetch original chunks
   - Return chunks with matched clause information

Benefits:
- Agent always sees ALL clauses before deciding
- Precise retrieval based on clause relevance
- Source tracking enables accurate chunk retrieval
- Sub-clause relationships preserved
"""


def create_root_clause_retriever(
    collection_name: str,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None
) -> RootClauseRetriever:
    """Factory function to create a RootClauseRetriever instance."""
    return RootClauseRetriever(
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
