"""
Clause-Aware Retrieval Service

This module provides intelligent retrieval that understands the chunk schema
and can dynamically build filters based on query intent.

Features:
- Schema awareness (knows available clause types, risk levels, layers)
- Dynamic filter generation based on query analysis
- Hybrid search combining semantic + clause-aware filtering
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from utils.logger import get_logger

logger = get_logger("ClauseAwareRetrieval")


# Schema definition - what the agent knows about the chunk structure
CHUNK_SCHEMA = {
    "clause_types": [
        "termination", "liability", "indemnification", "payment", "compensation",
        "confidentiality", "non_compete", "non_solicitation", "intellectual_property",
        "dispute_resolution", "governing_law", "force_majeure", "warranty",
        "limitation_of_liability", "assignment", "notice", "amendment", "severability",
        "employment", "duties", "probation", "benefits", "expenses", "leave",
        "working_hours", "overtime", "bonus", "equity", "stock_options", "other"
    ],
    "risk_levels": ["critical", "high", "medium", "low", "no_risk"],
    "layers": {
        1: "Raw text chunks (original document)",
        2: "Summaries with clause extraction and risk analysis",
        3: "Root summary (document corpus)"
    },
    "filterable_fields": {
        "layer": "integer - chunk layer (1, 2, 3+)",
        "risk_level": "string - risk assessment level",
        "clause_type": "string - type of legal clause (in clauses array)",
        "chunk_id": "string - unique chunk identifier"
    }
}

# Query patterns for intent detection
QUERY_PATTERNS = {
    "clause_types": {
        "payment": ["payment", "pay", "salary", "compensation", "remuneration", "wage", "bonus"],
        "termination": ["termination", "terminate", "end", "exit", "resignation", "dismissal", "fire"],
        "confidentiality": ["confidential", "confidentiality", "nda", "non-disclosure", "secret", "proprietary"],
        "liability": ["liability", "liable", "responsible", "responsibility", "damage"],
        "indemnification": ["indemnify", "indemnification", "indemnity", "hold harmless"],
        "non_compete": ["non-compete", "non compete", "compete", "competition", "competitor"],
        "intellectual_property": ["intellectual property", "ip", "patent", "copyright", "trademark", "invention"],
        "dispute_resolution": ["dispute", "arbitration", "mediation", "litigation", "court"],
        "governing_law": ["governing law", "jurisdiction", "applicable law", "venue"],
        "force_majeure": ["force majeure", "act of god", "unforeseen", "pandemic"],
        "warranty": ["warranty", "guarantee", "representation"],
        "assignment": ["assignment", "assign", "transfer", "successor"],
        "notice": ["notice", "notification", "inform", "notify"],
        "benefits": ["benefits", "insurance", "health", "medical", "dental", "vision"],
        "leave": ["leave", "vacation", "holiday", "sick", "maternity", "paternity"],
        "probation": ["probation", "probationary", "trial period"],
        "duties": ["duties", "responsibilities", "role", "job description", "scope"],
        "equity": ["equity", "stock", "shares", "options", "vesting", "esop"]
    },
    "risk_levels": {
        "critical": ["critical", "very high risk", "extremely risky", "dangerous", "severe"],
        "high": ["high risk", "risky", "concerning", "problematic", "unfavorable"],
        "medium": ["medium risk", "moderate", "some risk", "potential issue"],
        "low": ["low risk", "minor", "acceptable", "standard"],
        "no_risk": ["no risk", "safe", "favorable", "beneficial"]
    }
}


class ClauseAwareRetriever:
    """
    Intelligent retriever that understands chunk schema and builds dynamic filters.
    """
    
    def __init__(
        self,
        collection_name: str,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        llm_service = None
    ):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or config.QDRANT_URL
        self.qdrant_api_key = qdrant_api_key or config.QDRANT_API_KEY
        self.llm_service = llm_service
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key if self.qdrant_api_key else None
        )
        
        self.schema = CHUNK_SCHEMA
        self.query_patterns = QUERY_PATTERNS
        
        logger.info(f"ClauseAwareRetriever initialized for collection: {collection_name}")
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to detect intent and extract filter parameters.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with detected intents and suggested filters
        """
        query_lower = query.lower()
        
        intent = {
            "detected_clause_types": [],
            "detected_risk_levels": [],
            "is_risk_query": False,
            "is_clause_query": False,
            "is_general_query": True,
            "suggested_filters": {}
        }
        
        # Detect clause types
        for clause_type, keywords in self.query_patterns["clause_types"].items():
            for keyword in keywords:
                if keyword in query_lower:
                    if clause_type not in intent["detected_clause_types"]:
                        intent["detected_clause_types"].append(clause_type)
                    intent["is_clause_query"] = True
                    intent["is_general_query"] = False
        
        # Detect risk levels
        for risk_level, keywords in self.query_patterns["risk_levels"].items():
            for keyword in keywords:
                if keyword in query_lower:
                    if risk_level not in intent["detected_risk_levels"]:
                        intent["detected_risk_levels"].append(risk_level)
                    intent["is_risk_query"] = True
                    intent["is_general_query"] = False
        
        # Check for risk-related keywords without specific level
        risk_keywords = ["risk", "risky", "dangerous", "concern", "issue", "problem", "unfavorable"]
        for keyword in risk_keywords:
            if keyword in query_lower and not intent["is_risk_query"]:
                intent["is_risk_query"] = True
                intent["is_general_query"] = False
                # Default to high/critical if asking about risks
                intent["detected_risk_levels"] = ["critical", "high", "medium"]
        
        # Build suggested filters
        if intent["detected_clause_types"]:
            intent["suggested_filters"]["clause_types"] = intent["detected_clause_types"]
        
        if intent["detected_risk_levels"]:
            intent["suggested_filters"]["risk_levels"] = intent["detected_risk_levels"]
        
        logger.info(f"Query intent analysis: {intent}")
        return intent
    
    def get_available_values(self) -> Dict[str, Any]:
        """
        Get available filter values from the collection.
        This helps the agent understand what values exist.
        
        Returns:
            Dictionary of available values for each filterable field
        """
        try:
            # Get sample points to understand available values
            results = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            available = {
                "clause_types": set(),
                "risk_levels": set(),
                "layers": set()
            }
            
            for point in results[0]:
                payload = point.payload
                
                # Collect risk levels
                if payload.get("risk_level"):
                    available["risk_levels"].add(payload["risk_level"])
                
                # Collect layers
                if payload.get("layer"):
                    available["layers"].add(payload["layer"])
                
                # Collect clause types from clauses array
                for clause in payload.get("clauses", []):
                    if clause.get("clause_type"):
                        available["clause_types"].add(clause["clause_type"])
            
            # Convert sets to sorted lists
            result = {
                "clause_types": sorted(list(available["clause_types"])),
                "risk_levels": sorted(list(available["risk_levels"])),
                "layers": sorted(list(available["layers"]))
            }
            
            logger.info(f"Available values in collection: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting available values: {e}")
            return {"clause_types": [], "risk_levels": [], "layers": []}
    
    def build_clause_filter(
        self,
        clause_types: Optional[List[str]] = None,
        risk_levels: Optional[List[str]] = None,
        min_layer: int = 2,
        skip_risk_filter: bool = False
    ) -> Filter:
        """
        Build a Qdrant filter based on clause types and risk levels.
        
        Args:
            clause_types: List of clause types to filter for
            risk_levels: List of risk levels to filter for
            min_layer: Minimum layer to include (default: 2, excludes L1)
            skip_risk_filter: Skip risk_level filter (use if index not available)
            
        Returns:
            Qdrant Filter object
        """
        must_conditions = []
        
        # Always filter by minimum layer (layer index should exist)
        must_conditions.append(
            FieldCondition(
                key="layer",
                range=Range(gte=min_layer)
            )
        )
        
        # NOTE: risk_level filter requires a keyword index on the field
        # If index doesn't exist, we skip this filter and do post-filtering
        # The risk_level filtering will be done in post-processing instead
        
        filter_obj = Filter(
            must=must_conditions if must_conditions else None
        )
        
        logger.info(f"Built clause filter: min_layer={min_layer}, "
                   f"clause_types={clause_types}, risk_levels={risk_levels} (post-filter)")
        return filter_obj
    
    def search_with_clause_awareness(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform clause-aware search using query intent analysis.
        
        Args:
            query_vector: Query embedding vector
            query_text: Original query text for intent analysis
            top_k: Number of results to return
            
        Returns:
            List of search results with clause-aware filtering
        """
        # Analyze query intent
        intent = self.analyze_query_intent(query_text)
        
        # Build filter based on intent (layer filter only, risk filtering done post-search)
        clause_filter = self.build_clause_filter(
            clause_types=intent["suggested_filters"].get("clause_types"),
            risk_levels=intent["suggested_filters"].get("risk_levels"),
            min_layer=2
        )
        
        try:
            # Perform filtered search (fetch more to allow for post-filtering)
            fetch_limit = top_k * 3 if intent["is_risk_query"] else top_k
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=clause_filter,
                limit=fetch_limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            target_risk_levels = intent["detected_risk_levels"]
            
            for hit in search_results:
                # Check if clauses match the detected clause types
                matching_clauses = []
                if intent["detected_clause_types"]:
                    for clause in hit.payload.get("clauses", []):
                        clause_type = clause.get("clause_type", "").lower()
                        for detected_type in intent["detected_clause_types"]:
                            if detected_type in clause_type or clause_type in detected_type:
                                matching_clauses.append(clause)
                
                # Post-filter by risk level if specified
                chunk_risk = hit.payload.get("risk_level", "no_risk")
                risk_match = True
                if target_risk_levels:
                    risk_match = chunk_risk in target_risk_levels
                
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "chunk_id": hit.payload.get("chunk_id"),
                    "layer": hit.payload.get("layer"),
                    "text": hit.payload.get("text"),
                    "clauses": hit.payload.get("clauses", []),
                    "matching_clauses": matching_clauses,
                    "risk_level": chunk_risk,
                    "risk_match": risk_match,
                    "short_summary": hit.payload.get("short_summary"),
                    "metadata": hit.payload.get("metadata", {}),
                    "intent_match": {
                        "clause_types": intent["detected_clause_types"],
                        "risk_levels": intent["detected_risk_levels"],
                        "is_clause_query": intent["is_clause_query"],
                        "is_risk_query": intent["is_risk_query"]
                    }
                }
                results.append(result)
            
            # Sort by risk match first (matching risks come first), then by score
            if target_risk_levels:
                results.sort(key=lambda x: (not x.get("risk_match", False), -x.get("score", 0)))
            
            # Limit to top_k
            results = results[:top_k]
            
            logger.info(f"Clause-aware search returned {len(results)} results "
                       f"(risk_filter: {target_risk_levels}, clause_filter: {intent['detected_clause_types']})")
            return results
            
        except Exception as e:
            logger.error(f"Error in clause-aware search: {e}")
            raise
    
    def get_schema_description(self) -> str:
        """
        Get a human-readable description of the chunk schema.
        Useful for LLM agents to understand available filters.
        
        Returns:
            Schema description string
        """
        desc = """
CHUNK SCHEMA FOR LEGAL DOCUMENT RAG:

1. CLAUSE TYPES (filterable):
   - payment, compensation, salary, bonus
   - termination, resignation, dismissal
   - confidentiality, non-disclosure
   - liability, indemnification
   - non_compete, non_solicitation
   - intellectual_property
   - dispute_resolution, arbitration
   - governing_law, jurisdiction
   - force_majeure
   - warranty, guarantee
   - assignment, transfer
   - notice, notification
   - benefits, insurance, leave
   - probation, duties
   - equity, stock_options
   - other

2. RISK LEVELS (filterable):
   - critical: Severe unfavorable terms
   - high: Significant concerns
   - medium: Moderate issues
   - low: Minor concerns
   - no_risk: Favorable/standard terms

3. LAYERS:
   - Layer 1: Raw text chunks (excluded from search)
   - Layer 2: Summaries with extracted clauses and risk analysis
   - Layer 3+: Aggregated summaries (document corpus)

4. FILTERABLE FIELDS:
   - layer (integer): Chunk layer number
   - risk_level (string): Risk assessment
   - clauses (array): Contains clause_type, content, importance
"""
        return desc


def create_clause_aware_retriever(
    collection_name: str,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None
) -> ClauseAwareRetriever:
    """
    Factory function to create a ClauseAwareRetriever instance.
    
    Args:
        collection_name: Name of the Qdrant collection
        qdrant_url: Optional Qdrant URL
        qdrant_api_key: Optional Qdrant API key
        
    Returns:
        ClauseAwareRetriever instance
    """
    return ClauseAwareRetriever(
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
