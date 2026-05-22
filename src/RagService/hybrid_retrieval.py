"""
Hybrid Retrieval Strategy

Combines LLM-based clause selection with vector similarity search for optimal retrieval.

Architecture:
1. Phase 1: Load clause index from root chunk
2. Phase 2A: LLM selects relevant clauses (parallel)
3. Phase 2B: Vector similarity search (parallel)
4. Phase 3: Merge, deduplicate, and rank by hybrid score

Hybrid Score = α×LLM + β×vector + γ×importance
Where: α=0.6, β=0.3, γ=0.1
"""

import os
import sys
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from utils.logger import get_logger
from prompts import (
    LLM_CLAUSE_SELECTOR_SYSTEM_PROMPT,
    ANSWER_GENERATION_SYSTEM_PROMPT,
    get_clause_selection_prompt,
    get_answer_generation_prompt,
    get_large_doc_answer_generation_prompt,
)

# Import LLMService
try:
    from LLMService.LLMService import LLMService
except ImportError:
    try:
        from services.LLMService import LLMService
    except ImportError:
        from llm_service import LLMService

from embedding_service import OpenAIEmbeddings

logger = get_logger("HybridRetrieval")


@dataclass
class RankedChunk:
    """A chunk with hybrid ranking score."""
    chunk_id: str
    layer: int
    text: str
    clauses: List[Dict[str, Any]]
    hybrid_score: float
    llm_score: float = 0.0
    vector_score: float = 0.0
    importance_score: float = 0.0
    matched_clause_ids: List[str] = field(default_factory=list)
    source: str = "unknown"  # "llm", "vector", or "both"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "layer": self.layer,
            "text": self.text,
            "short_summary": self.metadata.get("short_summary", ""),
            "clauses": self.clauses,
            "hybrid_score": self.hybrid_score,
            "llm_score": self.llm_score,
            "vector_score": self.vector_score,
            "importance_score": self.importance_score,
            "matched_clause_ids": self.matched_clause_ids,
            "source": self.source,
            "metadata": self.metadata,
        }


class HybridRetriever:
    """
    Hybrid retrieval combining LLM clause selection + vector similarity.
    
    Flow:
    1. Get clause index from root chunk (Layer 8)
    2. Run LLM clause selection and vector search in parallel
    3. Merge results with hybrid scoring
    """
    
    def __init__(
        self,
        collection_name: str,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        llm_service: Optional[LLMService] = None,
        embeddings: Optional[OpenAIEmbeddings] = None
    ):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or config.QDRANT_URL
        self.qdrant_api_key = qdrant_api_key or config.QDRANT_API_KEY
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key if self.qdrant_api_key else None
        )
        
        self.llm_service = llm_service or LLMService(model=config.RETRIEVAL_LLM_MODEL)
        self.embeddings = embeddings or OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
        
        # Hybrid scoring weights (standard mode)
        self.llm_weight = config.HYBRID_LLM_WEIGHT
        self.vector_weight = config.HYBRID_VECTOR_WEIGHT
        self.importance_weight = config.HYBRID_IMPORTANCE_WEIGHT
        self.vector_top_k = config.HYBRID_VECTOR_TOP_K
        self.min_vector_score = config.HYBRID_MIN_VECTOR_SCORE
        self._fusion_override: Optional[Tuple[float, float, float]] = None
        
        # Cache
        self._root_chunk_cache = None
        self._clause_index_cache = None
        self._clause_type_counts_cache: Optional[Dict[str, int]] = None
        
        logger.info(
            f"HybridRetriever initialized for collection: {collection_name} "
            f"(retrieval_llm={config.RETRIEVAL_LLM_MODEL})"
        )
        logger.info(f"Weights: LLM={self.llm_weight}, Vector={self.vector_weight}, Importance={self.importance_weight}")

    def _scroll_all_points(
        self,
        scroll_filter: Optional[Filter] = None,
    ) -> List[Any]:
        """Paginate Qdrant scroll to avoid missing points beyond a single page."""
        all_points = []
        offset = None
        page_size = config.QDRANT_SCROLL_PAGE_SIZE
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(batch)
            if offset is None:
                break
        return all_points

    def _fetch_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch chunk payloads by chunk_id using a filtered scroll."""
        if not chunk_ids:
            return []
        try:
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="chunk_id",
                        match=MatchAny(any=list(chunk_ids)),
                    )
                ]
            )
            points = self._scroll_all_points(scroll_filter=scroll_filter)
            return [p.payload for p in points if p.payload.get("chunk_id")]
        except Exception as e:
            logger.error(f"Error fetching chunks by id: {e}")
            return []

    @staticmethod
    def _build_clause_type_histogram(
        clause_index: Dict[str, Dict[str, Any]],
        clause_type_counts: Optional[Dict[str, int]] = None,
    ) -> str:
        """Build a compact type histogram without enumerating every clause ID."""
        if clause_type_counts:
            counts = clause_type_counts
        else:
            counts: Dict[str, int] = {}
            for info in clause_index.values():
                ctype = info.get("clause_type", "Unknown")
                counts[ctype] = counts.get(ctype, 0) + 1
        lines = [
            f"- {ctype}: {count} provision(s)"
            for ctype, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ]
        return "\n".join(lines) if lines else "No clause type breakdown available."

    @staticmethod
    def _is_broad_query(query: str) -> bool:
        """Detect queries that benefit from wider vector recall."""
        q = query.lower()
        broad_signals = (
            "how many clause",
            "list all",
            "all clause",
            "summarize",
            "summary",
            "overview",
            "what is the contract about",
            "entire agreement",
            "whole contract",
        )
        return any(s in q for s in broad_signals)

    def _set_fusion_weights(self, llm_w: float, vector_w: float, importance_w: float) -> None:
        self._fusion_override = (llm_w, vector_w, importance_w)

    def _clear_fusion_weights(self) -> None:
        self._fusion_override = None

    def _get_fusion_weights(self) -> Tuple[float, float, float]:
        if self._fusion_override:
            return self._fusion_override
        return self.llm_weight, self.vector_weight, self.importance_weight
    
    async def get_root_chunk(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Find and return the root chunk (highest layer)."""
        if self._root_chunk_cache and not force_refresh:
            return self._root_chunk_cache
        
        try:
            all_points = self._scroll_all_points()
            if not all_points:
                logger.warning(f"No chunks found in collection: {self.collection_name}")
                return None
            
            max_layer = max((p.payload.get("layer", 0) for p in all_points), default=0)
            logger.info(f"Found max layer: {max_layer}")
            
            root_chunks = [
                p for p in all_points
                if p.payload.get("layer") == max_layer
            ]
            
            if not root_chunks:
                logger.warning(f"No root chunks found at layer {max_layer}")
                return None
            
            self._root_chunk_cache = root_chunks[0].payload
            meta = self._root_chunk_cache.get("metadata") or {}
            type_counts = (
                self._root_chunk_cache.get("clause_type_counts")
                or meta.get("clause_type_counts")
            )
            if type_counts:
                self._clause_type_counts_cache = type_counts
            logger.info(
                f"Root chunk: {self._root_chunk_cache.get('chunk_id')} "
                f"(layer {max_layer}, {len(self._root_chunk_cache.get('clauses', []))} clauses)"
            )
            return self._root_chunk_cache
            
        except Exception as e:
            logger.error(f"Error finding root chunk: {e}")
            return None
    
    async def get_clause_index(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get clause index from MongoDB (large docs) or root chunk."""
        if self._clause_index_cache and not force_refresh:
            return self._clause_index_cache

        # MongoDB first for collections with persisted large indices
        try:
            from services.mongodb_service import MongoDBService
            mongo = MongoDBService()
            stored = await mongo.get_stored_clause_index(self.collection_name)
            if stored and stored.get("clause_index"):
                self._clause_index_cache = stored["clause_index"]
                self._clause_type_counts_cache = stored.get("clause_type_counts")
                logger.info(
                    f"Clause index loaded from MongoDB: {len(self._clause_index_cache)} clauses"
                )
                return self._clause_index_cache
        except Exception as e:
            logger.debug(f"MongoDB clause index not used: {e}")

        root_chunk = await self.get_root_chunk(force_refresh)
        if not root_chunk:
            return {}

        self._clause_index_cache = root_chunk.get("clause_index", {}) or {}
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
                        "content": clause.get("content", "")[:200],
                    }

        if not self._clause_type_counts_cache:
            meta = root_chunk.get("metadata") or {}
            self._clause_type_counts_cache = (
                root_chunk.get("clause_type_counts") or meta.get("clause_type_counts")
            )

        logger.info(f"Clause index loaded: {len(self._clause_index_cache)} clauses")
        return self._clause_index_cache
    
    async def _llm_select_clauses(
        self,
        query: str,
        clause_index: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Use LLM to select relevant clauses from the index.
        
        Returns:
            Tuple of (selected_clause_ids, chunks_from_llm)
        """
        import re
        
        # Format clause list for LLM - include content preview for better matching
        clause_list = []
        for clause_id, info in clause_index.items():
            content_preview = info.get("content", "")[:100]
            clause_list.append(
                f"- {clause_id}: {info.get('clause_type', 'Unknown')} "
                f"[{info.get('importance', 'medium')}] - {content_preview}"
            )
        
        clauses_text = "\n".join(clause_list[:500])  # Limit to avoid token overflow
        
        prompt = get_clause_selection_prompt(query, clauses_text)

        selected_clause_ids = []
        
        try:
            messages = [
                {"role": "system", "content": LLM_CLAUSE_SELECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_service.chat_completion(messages, max_tokens=500, temperature=0.1)
            
            if response and response.strip():
                response_clean = response.strip()
                
                # Remove markdown code blocks
                if '```' in response_clean:
                    response_clean = re.sub(r'```json?\s*', '', response_clean)
                    response_clean = re.sub(r'\s*```', '', response_clean)
                    response_clean = response_clean.strip()
                
                # Extract JSON array
                array_match = re.search(r'\[.*?\]', response_clean, re.DOTALL)
                if array_match:
                    response_clean = array_match.group(0)
                
                parsed = json.loads(response_clean)
                if isinstance(parsed, list):
                    selected_clause_ids = [c for c in parsed if c in clause_index]
                    
        except Exception as e:
            logger.warning(f"LLM clause selection failed: {e}, using keyword fallback")
            selected_clause_ids = self._keyword_clause_match(query, clause_index)
        
        if not selected_clause_ids:
            selected_clause_ids = self._keyword_clause_match(query, clause_index)
        
        logger.info(f"LLM selected {len(selected_clause_ids)} clauses")
        
        # Get chunks for selected clauses
        chunks = await self._get_chunks_by_clause_ids(selected_clause_ids, clause_index)
        
        return selected_clause_ids, chunks
    
    def _keyword_clause_match(
        self,
        query: str,
        clause_index: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Fallback keyword-based clause matching."""
        import re
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Common legal keyword mappings
        keyword_mappings = {
            "termination": ["termination", "end", "resignation", "dismissal", "notice"],
            "salary": ["salary", "compensation", "remuneration", "pay", "wage"],
            "confidentiality": ["confidential", "non-disclosure", "nda", "secret"],
            "intellectual": ["intellectual", "property", "ip", "invention", "patent"],
            "non-compete": ["non-compete", "competition", "restrictive"],
            "leave": ["leave", "vacation", "holiday", "absence"],
            "benefits": ["benefits", "insurance", "health", "medical"],
            "dispute": ["dispute", "arbitration", "resolution", "mediation"],
        }
        
        search_terms = set()
        for word in query_words:
            if word in keyword_mappings:
                search_terms.update(keyword_mappings[word])
            else:
                search_terms.add(word)
        
        matched = []
        for clause_id, info in clause_index.items():
            clause_type = info.get("clause_type", "").lower()
            content = info.get("content", "").lower()
            
            for term in search_terms:
                if term in clause_type or term in content:
                    matched.append(clause_id)
                    break
        
        # Fallback to high-importance clauses
        if not matched:
            matched = [
                cid for cid, info in clause_index.items()
                if info.get("importance") == "high" and info.get("parent_clause_id") is None
            ][:10]
        
        return matched
    
    async def _get_chunks_by_clause_ids(
        self,
        clause_ids: List[str],
        clause_index: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks for selected clause IDs."""
        # Get unique source chunk IDs
        source_chunk_ids = set()
        for clause_id in clause_ids:
            if clause_id in clause_index:
                source_id = clause_index[clause_id].get("source_chunk_id", "")
                if source_id:
                    source_chunk_ids.add(source_id)
        
        if not source_chunk_ids:
            return []

        chunks = []
        try:
            for payload in self._fetch_chunks_by_ids(list(source_chunk_ids)):
                chunk_id = payload.get("chunk_id", "")
                matched_clauses = [
                    cid for cid in clause_ids
                    if clause_index.get(cid, {}).get("source_chunk_id") == chunk_id
                ]
                chunks.append({
                    **payload,
                    "matched_clause_ids": matched_clauses,
                    "source": "llm",
                })
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")

        return chunks
    
    async def _vector_search(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        top_k = top_k or self.vector_top_k
        
        try:
            # Get query embedding
            query_vector = await self.embeddings.async_embed_query(query)
            
            # Search in Qdrant (exclude Layer 1 raw chunks)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="layer",
                            range=Range(gte=2)  # Only L2+ chunks
                        )
                    ]
                ),
                with_payload=True
            )
            
            chunks = []
            for result in results:
                if result.score >= self.min_vector_score:
                    chunk_data = {
                        **result.payload,
                        "vector_score": result.score,
                        "source": "vector"
                    }
                    chunks.append(chunk_data)
            
            logger.info(f"Vector search returned {len(chunks)} chunks (min_score={self.min_vector_score})")
            return chunks
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _compute_importance_score(self, chunk: Dict[str, Any]) -> float:
        """Compute importance score for a chunk based on its clauses."""
        clauses = chunk.get("clauses", [])
        if not clauses:
            return 0.0
        
        importance_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
        
        total_importance = sum(
            importance_map.get(c.get("importance", "medium"), 0.5)
            for c in clauses
        )
        
        # Normalize by number of clauses
        return min(total_importance / len(clauses), 1.0)
    
    def _fuse_and_rank(
        self,
        llm_chunks: List[Dict[str, Any]],
        vector_chunks: List[Dict[str, Any]],
        clause_ids_from_llm: List[str],
        top_k: int
    ) -> List[RankedChunk]:
        """Merge and rank chunks using hybrid scoring."""
        
        # Build chunk map for deduplication
        chunk_map: Dict[str, Dict[str, Any]] = {}
        
        # Add LLM chunks
        for chunk in llm_chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id:
                chunk_map[chunk_id] = {
                    **chunk,
                    "llm_selected": True,
                    "vector_score": 0.0
                }
        
        # Add/merge vector chunks
        for chunk in vector_chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id:
                if chunk_id in chunk_map:
                    # Merge - chunk found by both methods
                    chunk_map[chunk_id]["vector_score"] = chunk.get("vector_score", 0.0)
                    chunk_map[chunk_id]["source"] = "both"
                else:
                    chunk_map[chunk_id] = {
                        **chunk,
                        "llm_selected": False,
                        "vector_score": chunk.get("vector_score", 0.0)
                    }
        
        # Compute hybrid scores and create RankedChunk objects
        ranked_chunks = []
        for chunk_id, chunk in chunk_map.items():
            llm_score = 1.0 if chunk.get("llm_selected", False) else 0.0
            vector_score = chunk.get("vector_score", 0.0)
            importance_score = self._compute_importance_score(chunk)
            
            llm_w, vector_w, importance_w = self._get_fusion_weights()
            hybrid_score = (
                llm_w * llm_score +
                vector_w * vector_score +
                importance_w * importance_score
            )
            
            metadata = dict(chunk.get("metadata") or {})
            if chunk.get("short_summary") and "short_summary" not in metadata:
                metadata["short_summary"] = chunk.get("short_summary")

            ranked_chunk = RankedChunk(
                chunk_id=chunk_id,
                layer=chunk.get("layer", 0),
                text=chunk.get("text", ""),
                clauses=chunk.get("clauses", []),
                hybrid_score=hybrid_score,
                llm_score=llm_score,
                vector_score=vector_score,
                importance_score=importance_score,
                matched_clause_ids=chunk.get("matched_clause_ids", []),
                source=chunk.get("source", "unknown"),
                metadata=metadata,
            )
            ranked_chunks.append(ranked_chunk)
        
        # Sort by hybrid score descending
        ranked_chunks.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        logger.info(f"Fused {len(ranked_chunks)} unique chunks, returning top {top_k}")
        
        return ranked_chunks[:top_k]
    
    async def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        clause_index: Dict[str, Dict[str, Any]] = None,
        total_clauses: int = 0,
        large_doc_mode: bool = False,
        clause_type_counts: Optional[Dict[str, int]] = None,
    ) -> str:
        """
        Generate a final answer using LLM based on retrieved chunks AND full clause index.
        
        Args:
            query: User query
            chunks: Retrieved chunks with text and clause information
            clause_index: Full clause index from root chunk (for accurate counts)
            total_clauses: Total number of clauses in the document
            
        Returns:
            LLM-generated answer string
        """
        if not chunks:
            return "No relevant information found to answer your question."
        
        clause_summary = ""
        if clause_index and total_clauses > 0:
            if large_doc_mode or total_clauses > config.HYBRID_LLM_CLAUSE_THRESHOLD:
                histogram = self._build_clause_type_histogram(
                    clause_index,
                    clause_type_counts=clause_type_counts or self._clause_type_counts_cache,
                )
                clause_summary = f"""
DOCUMENT CLAUSE SUMMARY (FULL DOCUMENT - {total_clauses} total indexed provisions):
{histogram}
"""
            else:
                clause_types: Dict[str, List[str]] = {}
                for clause_id, info in clause_index.items():
                    ctype = info.get("clause_type", "Unknown")
                    clause_types.setdefault(ctype, []).append(clause_id)
                clause_summary = f"""
DOCUMENT CLAUSE SUMMARY (FULL DOCUMENT - {total_clauses} total clauses):
"""
                for ctype, clause_ids in sorted(clause_types.items()):
                    clause_summary += (
                        f"- {ctype}: {len(clause_ids)} clauses "
                        f"({', '.join(clause_ids[:5])}{'...' if len(clause_ids) > 5 else ''})\n"
                    )
        
        # Document-level summary from chunk short_summaries
        doc_summaries = []
        for chunk in chunks:
            short_summary = chunk.get("short_summary") or chunk.get("metadata", {}).get("short_summary")
            if short_summary and short_summary not in doc_summaries:
                doc_summaries.append(short_summary)
        document_summary_block = ""
        if doc_summaries:
            document_summary_block = "DOCUMENT OVERVIEW (from hierarchical summaries):\n" + "\n".join(
                f"- {s[:500]}" for s in doc_summaries[:5]
            ) + "\n\n"

        # Build context from retrieved chunks (summaries first, excerpts for citation)
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get("text", "")
            short_summary = chunk.get("short_summary") or chunk.get("metadata", {}).get("short_summary", "")
            chunk_id = chunk.get("chunk_id", f"Chunk {i}")
            clauses = chunk.get("clauses", [])

            clause_info = []
            for c in clauses[:5]:
                clause_info.append(f"- {c.get('clause_id', 'N/A')}: {c.get('clause_type', 'Unknown')}")
            clause_str = "\n".join(clause_info) if clause_info else "No specific clauses"

            summary_section = short_summary if short_summary else chunk_text[:600]
            excerpt = chunk_text[:400] + ("..." if len(chunk_text) > 400 else "") if short_summary else ""

            context_parts.append(f"""
--- {chunk_id} ---
Summary: {summary_section}
{f'Supporting excerpt: {excerpt}' if excerpt else ''}
Clauses in this section:
{clause_str}
""")

        context = document_summary_block + "\n".join(context_parts)

        use_large_prompt = large_doc_mode or (
            total_clauses > config.HYBRID_LLM_CLAUSE_THRESHOLD
        )
        if use_large_prompt:
            histogram = self._build_clause_type_histogram(
                clause_index or {},
                clause_type_counts=clause_type_counts or self._clause_type_counts_cache,
            )
            prompt = get_large_doc_answer_generation_prompt(
                query, histogram, context, total_clauses
            )
        else:
            prompt = get_answer_generation_prompt(query, clause_summary, context, total_clauses)

        try:
            messages = [
                {"role": "system", "content": ANSWER_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_service.chat_completion(
                messages=messages,
                max_tokens=8000,
                temperature=0.45,
            )
            
            if response and response.strip():
                logger.info(f"Generated answer: {len(response)} chars")
                return response.strip()
            else:
                return "Unable to generate an answer. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        generate_answer: bool = False
    ) -> Dict[str, Any]:
        """
        Main hybrid retrieval method.
        
        Args:
            query: User query
            top_k: Number of chunks to return
            generate_answer: Whether to generate LLM answer from chunks
            
        Returns:
            Dictionary with retrieval results
        """
        logger.info(f"Hybrid retrieval for: {query[:50]}...")
        
        # Phase 1: Get clause index
        clause_index = await self.get_clause_index()
        if not clause_index:
            logger.warning("No clause index found, falling back to vector-only search")
            vector_chunks = await self._vector_search(query, top_k)
            chunk_dicts = [self._chunk_to_dict(c) for c in vector_chunks]
            result = {
                "status": "partial",
                "message": "No clause index found, used vector search only",
                "chunks": chunk_dicts,
                "num_chunks": len(chunk_dicts),
                "clause_index": {},
                "total_clauses_available": 0,
                "retrieval_mode": "vector_only",
                "clause_count": 0,
            }
            if generate_answer and chunk_dicts:
                logger.info("Generating LLM answer from vector search (no clause index)...")
                result["answer"] = await self.generate_answer(
                    query=query,
                    chunks=chunk_dicts,
                    clause_index={},
                    total_clauses=0,
                )
            return result
        
        clause_count = len(clause_index)
        vector_top_k = (
            config.HYBRID_VECTOR_TOP_K_LARGE
            if clause_count > config.HYBRID_LLM_CLAUSE_THRESHOLD or self._is_broad_query(query)
            else self.vector_top_k
        )

        if clause_count > config.HYBRID_LLM_CLAUSE_THRESHOLD:
            retrieval_mode = "vector_only"
            logger.info(
                f"Large document ({clause_count} clauses > {config.HYBRID_LLM_CLAUSE_THRESHOLD}): "
                "skipping LLM clause selection, using vector-first retrieval"
            )
            self._set_fusion_weights(
                config.HYBRID_LLM_WEIGHT_LARGE,
                config.HYBRID_VECTOR_WEIGHT_LARGE,
                config.HYBRID_IMPORTANCE_WEIGHT_LARGE,
            )
            vector_chunks = await self._vector_search(query, top_k=vector_top_k)
            clause_ids, llm_chunks = [], []
        else:
            retrieval_mode = "hybrid"
            self._clear_fusion_weights()
            llm_task = self._llm_select_clauses(query, clause_index)
            vector_task = self._vector_search(query, top_k=vector_top_k)
            (clause_ids, llm_chunks), vector_chunks = await asyncio.gather(
                llm_task, vector_task
            )

        logger.info(f"LLM: {len(llm_chunks)} chunks, Vector: {len(vector_chunks)} chunks")

        try:
            ranked_chunks = self._fuse_and_rank(llm_chunks, vector_chunks, clause_ids, top_k)
        finally:
            self._clear_fusion_weights()

        llm_w, vector_w, importance_w = self._get_fusion_weights()
        if retrieval_mode == "vector_only":
            llm_w, vector_w, importance_w = (
                config.HYBRID_LLM_WEIGHT_LARGE,
                config.HYBRID_VECTOR_WEIGHT_LARGE,
                config.HYBRID_IMPORTANCE_WEIGHT_LARGE,
            )

        large_doc_mode = clause_count > config.HYBRID_LLM_CLAUSE_THRESHOLD
        result = {
            "status": "success",
            "query": query,
            "chunks": [rc.to_dict() for rc in ranked_chunks],
            "num_chunks": len(ranked_chunks),
            "llm_selected_clauses": clause_ids,
            "llm_chunks_count": len(llm_chunks),
            "vector_chunks_count": len(vector_chunks),
            "total_clauses_available": clause_count,
            "clause_index": clause_index,
            "retrieval_mode": retrieval_mode,
            "clause_count": clause_count,
            "scoring_weights": {
                "llm": llm_w,
                "vector": vector_w,
                "importance": importance_w,
            },
        }

        if generate_answer and ranked_chunks:
            logger.info(
                f"Generating LLM answer ({retrieval_mode}, {clause_count} clauses)..."
            )
            answer = await self.generate_answer(
                query=query,
                chunks=[rc.to_dict() for rc in ranked_chunks],
                clause_index=clause_index,
                total_clauses=clause_count,
                large_doc_mode=large_doc_mode,
                clause_type_counts=self._clause_type_counts_cache,
            )
            result["answer"] = answer

        return result
    
    def _chunk_to_dict(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Convert chunk to standard dict format."""
        return {
            "chunk_id": chunk.get("chunk_id", ""),
            "layer": chunk.get("layer", 0),
            "text": chunk.get("text", ""),
            "short_summary": chunk.get("short_summary", ""),
            "clauses": chunk.get("clauses", []),
            "hybrid_score": chunk.get("vector_score", 0.0),
            "source": "vector",
        }


def create_hybrid_retriever(
    collection_name: str,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None
) -> HybridRetriever:
    """Factory function to create a HybridRetriever instance."""
    return HybridRetriever(
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
