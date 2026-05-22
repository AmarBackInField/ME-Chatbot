"""
LangGraph-based Conversational Agent for Legal Document Analysis.
Routes natural language queries to appropriate agents.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Literal, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from utils.logger import get_logger
from llm_service import LLMService
from prompts import (
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    GENERAL_QUESTION_SYSTEM_PROMPT,
    LEGAL_ASSISTANT_CONVERSATIONAL_PROMPT,
    CANDEXAI_ASSISTANT_FALLBACK,
    is_meta_conversational_query,
    build_conversational_user_message,
    CLAUSE_EXTRACTION_FORMAT_PROMPT,
    RISK_ANALYSIS_FORMAT_PROMPT,
    COMPARISON_FORMAT_PROMPT,
    REPORT_GENERATION_FORMAT_PROMPT,
    GENERAL_FORMAT_PROMPT,
    HYBRID_ANSWER_FORMAT_PROMPT,
)

logger = get_logger("ConversationGraph")

# MongoDB checkpointer import (lazy loaded)
_mongodb_checkpointer = None


class ConversationState(TypedDict):
    """State for the conversation graph."""
    user_query: str
    document_text: str
    rag_context: Optional[str]
    intent: str
    agent_response: Dict[str, Any]
    final_response: str
    error: Optional[str]
    session_id: Optional[str]
    conversation_history: Optional[List[Dict[str, str]]]
    retrieval_metadata: Optional[Dict[str, Any]]


class IntentType:
    TECHNICAL_SPECS = "technical_specs"
    TROUBLESHOOTING = "troubleshooting"
    SAFETY_INFO = "safety_info"
    DESIGN_PARAMETERS = "design_parameters"
    GENERAL_QUESTION = "general_question"
    UNKNOWN = "unknown"


def get_mongodb_checkpointer() -> Optional[BaseCheckpointSaver]:
    """Get MongoDB checkpointer for LangGraph state persistence."""
    global _mongodb_checkpointer
    if _mongodb_checkpointer is not None:
        return _mongodb_checkpointer
    
    try:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        from pymongo import MongoClient
        
        client = MongoClient(
            config.MONGO_DB_URI,
            serverSelectionTimeoutMS=5000,
            maxPoolSize=200,
            minPoolSize=10
        )
        
        _mongodb_checkpointer = MongoDBSaver(
            client=client,
            db_name=config.MONGO_DB_NAME
        )
        logger.info("MongoDB checkpointer initialized for LangGraph")
        return _mongodb_checkpointer
    except ImportError as e:
        logger.warning(f"langgraph-checkpoint-mongodb not installed: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize MongoDB checkpointer: {e}")
        return None


class ConversationGraph:
    """
    LangGraph-based conversational interface for mechanical engineering document analysis.
    Routes natural language queries to appropriate handlers.
    Uses MongoDB for checkpointing and state persistence.
    """
    
    def __init__(self, use_checkpointer: bool = True):
        self.llm = LLMService()
        self.conversational_llm = LLMService(model=config.CONVERSATIONAL_LLM_MODEL)

        # Clause index cache for current session
        self._current_clause_index = None
        
        # MongoDB checkpointer for state persistence
        self._checkpointer = None
        self._use_checkpointer = use_checkpointer
        
        # Build the graph
        self.graph = self._build_graph()
        logger.info("ConversationGraph initialized for mechanical engineering domain")

    async def generate_conversational_response(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        collection_name: str = "",
        document_available: bool = False,
        document_name: str = "",
    ) -> str:
        """
        Meta chat: greetings, capabilities, memory. Uses fast model (not gpt-5).
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": LEGAL_ASSISTANT_CONVERSATIONAL_PROMPT},
        ]
        if conversation_history:
            for msg in conversation_history[-10:]:
                role = msg.get("role", "user")
                if role not in ("user", "assistant"):
                    continue
                content = (msg.get("content") or "").strip()
                if content:
                    messages.append({"role": role, "content": content[:2000]})

        messages.append({
            "role": "user",
            "content": build_conversational_user_message(
                user_query=user_query,
                collection_name=collection_name,
                document_available=document_available,
                document_name=document_name,
            ),
        })

        try:
            answer = await self.conversational_llm.chat_completion(
                messages=messages,
                max_tokens=1500,
                temperature=0.65,
            )
            if answer and answer.strip():
                return answer.strip()
        except Exception as e:
            logger.error(f"Conversational LLM failed: {e}")

        logger.warning("Conversational response empty; using CandexAI fallback")
        return CANDEXAI_ASSISTANT_FALLBACK
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("technical_specs", self._handle_technical_specs)
        workflow.add_node("troubleshooting", self._handle_troubleshooting)
        workflow.add_node("safety_info", self._handle_safety_info)
        workflow.add_node("design_parameters", self._handle_design_parameters)
        workflow.add_node("general_question", self._handle_general_question)
        workflow.add_node("format_response", self._format_response)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional edges based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                IntentType.TECHNICAL_SPECS: "technical_specs",
                IntentType.TROUBLESHOOTING: "troubleshooting",
                IntentType.SAFETY_INFO: "safety_info",
                IntentType.DESIGN_PARAMETERS: "design_parameters",
                IntentType.GENERAL_QUESTION: "general_question",
                IntentType.UNKNOWN: "general_question"
            }
        )
        
        # All handler nodes lead to format_response
        workflow.add_edge("technical_specs", "format_response")
        workflow.add_edge("troubleshooting", "format_response")
        workflow.add_edge("safety_info", "format_response")
        workflow.add_edge("design_parameters", "format_response")
        workflow.add_edge("general_question", "format_response")
        
        # format_response leads to END
        workflow.add_edge("format_response", END)
        
        # Compile with checkpointer if available
        if self._use_checkpointer:
            self._checkpointer = get_mongodb_checkpointer()
            if self._checkpointer:
                return workflow.compile(checkpointer=self._checkpointer)
        
        return workflow.compile()
    
    async def _classify_intent(self, state: ConversationState) -> ConversationState:
        """Classify the user's intent from their natural language query."""
        user_query = state["user_query"].lower()
        
        # Simple keyword-based classification for mechanical domain
        intent = IntentType.UNKNOWN
        
        specs_keywords = ["specification", "spec", "tolerance", "dimension", "material", "property", "torque", "pressure", "temperature", "size", "weight"]
        troubleshooting_keywords = ["error", "fault", "troubleshoot", "maintenance", "repair", "diagnose", "fix", "problem", "issue", "malfunction"]
        safety_keywords = ["safety", "warning", "hazard", "limit", "caution", "danger", "precaution", "protective"]
        design_keywords = ["design", "calculation", "standard", "compliance", "criteria", "requirement", "ISO", "ASME", "ANSI"]
        
        if any(kw in user_query for kw in specs_keywords):
            intent = IntentType.TECHNICAL_SPECS
        elif any(kw in user_query for kw in troubleshooting_keywords):
            intent = IntentType.TROUBLESHOOTING
        elif any(kw in user_query for kw in safety_keywords):
            intent = IntentType.SAFETY_INFO
        elif any(kw in user_query for kw in design_keywords):
            intent = IntentType.DESIGN_PARAMETERS
        else:
            # Use LLM for more complex intent classification
            intent = await self._llm_classify_intent(state["user_query"])
        
        logger.info(f"Classified intent: {intent} for query: {user_query[:50]}...")
        state["intent"] = intent
        return state
    
    async def _llm_classify_intent(self, query: str) -> str:
        """Use LLM to classify intent when keyword matching fails."""
        try:
            messages = [
                {"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
            
            # Use non-streaming for intent classification (streaming returns empty responses)
            logger.debug(f"Calling LLM with messages: {messages}")
            
            raw_intent = await self.llm.chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=100,
            )
            
            # Log raw response for debugging
            logger.info(f"LLM raw intent response: '{raw_intent}' (length: {len(raw_intent)}) for query: '{query[:50]}...'")
            
            if not raw_intent or not raw_intent.strip():
                logger.error(f"Empty response from LLM for intent classification. Messages sent: {messages[:1]}")
                return IntentType.GENERAL_QUESTION
            
            # Clean and normalize the response
            intent = raw_intent.lower().strip()
            
            # Remove any extra text (e.g., "The intent is: clause_extraction" -> "clause_extraction")
            if ":" in intent:
                intent = intent.split(":")[-1].strip()
            
            # Remove quotes if present
            intent = intent.replace('"', '').replace("'", "")
            
            # Check if it matches any valid intent
            valid_intents = [
                IntentType.TECHNICAL_SPECS,
                IntentType.TROUBLESHOOTING,
                IntentType.SAFETY_INFO,
                IntentType.DESIGN_PARAMETERS,
                IntentType.GENERAL_QUESTION
            ]
            
            if intent in valid_intents:
                logger.info(f"✅ LLM classified as: {intent}")
                return intent
            else:
                logger.warning(f"⚠️ LLM returned invalid intent: '{intent}', defaulting to general_question")
                return IntentType.GENERAL_QUESTION
                
        except Exception as e:
            logger.error(f"LLM intent classification failed: {e}")
            return IntentType.GENERAL_QUESTION
    
    def _route_by_intent(self, state: ConversationState) -> str:
        """Route to the appropriate agent based on intent."""
        return state.get("intent", IntentType.GENERAL_QUESTION)
    
    async def _handle_technical_specs(self, state: ConversationState) -> ConversationState:
        """Handle technical specification extraction requests with RAG context."""
        try:
            document_text = state.get("document_text", "")
            rag_context = state.get("rag_context", "")
            
            # OPTIMIZATION: If RAG context has hybrid retrieval answer, use it directly
            if rag_context and "[HYBRID RETRIEVAL ANALYSIS]" in rag_context:
                logger.info("⚡ SKIP: Using hybrid retrieval answer directly (no redundant spec extraction)")
                
                # Extract section IDs from the hybrid answer using regex
                import re
                section_ids = re.findall(r'L\d+_C\d+(?:_S\d+)*(?:\.\d+)?', rag_context)
                unique_section_ids = list(dict.fromkeys(section_ids))
                
                # Build sections list from extracted IDs
                sections = [{"section_id": sid, "source": "hybrid_retrieval"} for sid in unique_section_ids]
                
                logger.info(f"📋 Extracted {len(unique_section_ids)} unique section IDs from hybrid answer")
                
                parts = rag_context.split("[SUPPORTING DOCUMENT EXCERPTS]")
                draft_answer = parts[0].replace("[HYBRID RETRIEVAL ANALYSIS]", "").strip()
                document_section_count = len(self._current_clause_index) if self._current_clause_index else 0
                retrieval_meta = state.get("retrieval_metadata") or {}
                state["agent_response"] = {
                    "total_clauses": document_section_count or len(unique_section_ids),
                    "cited_clauses_count": len(unique_section_ids),
                    "document_clause_count": document_section_count,
                    "clauses": sections,
                    "clause_ids": unique_section_ids,
                    "document_type": "technical_document",
                    "hybrid_answer_used": True,
                    "draft_answer": draft_answer,
                    "retrieval_mode": retrieval_meta.get("retrieval_mode", "hybrid"),
                    "clause_count": retrieval_meta.get("clause_count", document_section_count),
                }
                return state
            
            if not document_text:
                if rag_context:
                    logger.info("No full document text; answering from RAG context")
                    answer = await self.llm.chat_completion(
                        messages=[
                            {"role": "system", "content": GENERAL_QUESTION_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": f"RETRIEVED TECHNICAL DOCUMENT CONTEXT:\n{rag_context}\n\nUser Question: {state.get('user_query', '')}",
                            },
                        ],
                        temperature=0.5,
                        max_tokens=2000,
                    )
                    state["agent_response"] = {
                        "answer": answer,
                        "total_clauses": 0,
                        "from_rag_context": True,
                    }
                    return state
                state["error"] = "No document text available for specification extraction"
                state["agent_response"] = {}
                return state
            
            # Use RAG context directly for answer generation
            if rag_context:
                logger.info("Using RAG context for technical specification extraction")
                answer = await self.llm.chat_completion(
                    messages=[
                        {"role": "system", "content": GENERAL_QUESTION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"TECHNICAL DOCUMENT CONTEXT:\n{rag_context}\n\nUser Question: {state.get('user_query', '')}\n\nExtract and present the relevant technical specifications with units and values.",
                        },
                    ],
                    temperature=0.5,
                    max_tokens=2000,
                )
                state["agent_response"] = {"answer": answer, "from_rag_context": True}
            else:
                state["agent_response"] = {"answer": "No technical specifications found in the document."}
            
            logger.info("Technical specification extraction completed")
        except Exception as e:
            logger.error(f"Technical specs extraction error: {e}")
            state["error"] = str(e)
            state["agent_response"] = {}
        return state
    
    async def _handle_safety_info(self, state: ConversationState) -> ConversationState:
        """Handle safety information extraction requests with RAG context."""
        try:
            document_text = state.get("document_text", "")
            rag_context = state.get("rag_context", "")
            
            # OPTIMIZATION: If RAG context has hybrid retrieval answer, use it directly
            if rag_context and "[HYBRID RETRIEVAL ANALYSIS]" in rag_context:
                logger.info("⚡ SKIP: Using hybrid retrieval answer directly (no redundant safety extraction)")
                
                # Extract section IDs and safety categories from hybrid answer
                import re
                section_ids = re.findall(r'L\d+_C\d+(?:_S\d+)*(?:\.\d+)?', rag_context)
                unique_section_ids = list(dict.fromkeys(section_ids))
                
                # Count safety severity mentioned
                safety_categories = []
                safety_keywords = {
                    "critical": "critical",
                    "danger": "high",
                    "warning": "medium",
                    "caution": "low"
                }
                for keyword, level in safety_keywords.items():
                    if keyword in rag_context.lower():
                        safety_categories.append(level)
                
                # Determine overall safety level
                if "critical" in safety_categories or "danger" in rag_context.lower():
                    overall_safety = "critical"
                elif "warning" in safety_categories:
                    overall_safety = "high"
                elif "caution" in safety_categories:
                    overall_safety = "medium"
                else:
                    overall_safety = "see_hybrid_answer"
                
                logger.info(f"📋 Extracted {len(unique_section_ids)} section IDs, safety level: {overall_safety}")
                
                state["agent_response"] = {
                    "overall_risk_level": overall_safety,
                    "risk_score": 0,
                    "total_risks": len(unique_section_ids),
                    "total_clauses": len(unique_section_ids),
                    "clause_ids": unique_section_ids,
                    "risks": [],
                    "hybrid_answer_used": True,
                    "rag_context": rag_context
                }
                return state
            
            if not document_text:
                if rag_context:
                    logger.info("No full document text; answering from RAG context")
                    answer = await self.llm.chat_completion(
                        messages=[
                            {"role": "system", "content": GENERAL_QUESTION_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": f"RETRIEVED TECHNICAL DOCUMENT CONTEXT:\n{rag_context}\n\nUser Question: {state.get('user_query', '')}\n\nExtract and present safety warnings, operating limits, and hazard information.",
                            },
                        ],
                        temperature=0.5,
                        max_tokens=2000,
                    )
                    state["agent_response"] = {"answer": answer, "from_rag_context": True}
                    return state
                state["error"] = "No document text available for safety information extraction"
                state["agent_response"] = {}
                return state
            
            # Use RAG context for safety information extraction
            if rag_context:
                logger.info("Using RAG context for safety information extraction")
                answer = await self.llm.chat_completion(
                    messages=[
                        {"role": "system", "content": GENERAL_QUESTION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"TECHNICAL DOCUMENT CONTEXT:\n{rag_context}\n\nUser Question: {state.get('user_query', '')}\n\nExtract and present safety warnings, operating limits, hazards, and protective measures with specific values and units.",
                        },
                    ],
                    temperature=0.5,
                    max_tokens=2000,
                )
                state["agent_response"] = {"answer": answer, "from_rag_context": True}
            else:
                state["agent_response"] = {"answer": "No safety information found in the document."}
            
            logger.info("Safety information extraction completed")
        except Exception as e:
            logger.error(f"Safety info extraction error: {e}")
            state["error"] = str(e)
            state["agent_response"] = {}
        return state
    
    async def _handle_troubleshooting(self, state: ConversationState) -> ConversationState:
        """Handle troubleshooting and maintenance procedure requests with RAG context."""
        try:
            document_text = state.get("document_text", "")
            rag_context = state.get("rag_context", "")
            
            # OPTIMIZATION: If RAG context has hybrid retrieval answer, use it directly
            if rag_context and "[HYBRID RETRIEVAL ANALYSIS]" in rag_context:
                logger.info("⚡ SKIP: Using hybrid retrieval answer directly for troubleshooting")
                
                parts = rag_context.split("[SUPPORTING DOCUMENT EXCERPTS]")
                draft_answer = parts[0].replace("[HYBRID RETRIEVAL ANALYSIS]", "").strip()
                
                state["agent_response"] = {
                    "hybrid_answer_used": True,
                    "draft_answer": draft_answer,
                    "rag_context": rag_context
                }
                return state
            
            if not document_text:
                if rag_context:
                    logger.info("No full document text; answering from RAG context")
                    answer = await self.llm.chat_completion(
                        messages=[
                            {"role": "system", "content": GENERAL_QUESTION_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": f"RETRIEVED TECHNICAL DOCUMENT CONTEXT:\n{rag_context}\n\nUser Question: {state.get('user_query', '')}\n\nProvide troubleshooting steps, maintenance procedures, or diagnostic guidance.",
                            },
                        ],
                        temperature=0.5,
                        max_tokens=2000,
                    )
                    state["agent_response"] = {"answer": answer, "from_rag_context": True}
                    return state
                state["error"] = "No document text available for troubleshooting guidance"
                state["agent_response"] = {}
                return state
            
            # Use RAG context for troubleshooting guidance
            if rag_context:
                logger.info("Using RAG context for troubleshooting guidance")
                answer = await self.llm.chat_completion(
                    messages=[
                        {"role": "system", "content": GENERAL_QUESTION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"TECHNICAL DOCUMENT CONTEXT:\n{rag_context}\n\nUser Question: {state.get('user_query', '')}\n\nProvide step-by-step troubleshooting guidance, maintenance procedures, or diagnostic steps with specific technical details.",
                        },
                    ],
                    temperature=0.5,
                    max_tokens=2000,
                )
                state["agent_response"] = {"answer": answer, "from_rag_context": True}
            else:
                state["agent_response"] = {"answer": "No troubleshooting information found in the document."}
            
            logger.info("Troubleshooting guidance completed")
        except Exception as e:
            logger.error(f"Troubleshooting error: {e}")
            state["error"] = str(e)
            state["agent_response"] = {}
        return state
    
    async def _handle_design_parameters(self, state: ConversationState) -> ConversationState:
        """Handle design parameters and standards compliance requests with RAG context."""
        try:
            document_text = state.get("document_text", "")
            rag_context = state.get("rag_context", "")
            
            # OPTIMIZATION: If RAG context has hybrid retrieval answer, use it directly
            if rag_context and "[HYBRID RETRIEVAL ANALYSIS]" in rag_context:
                logger.info("⚡ SKIP: Using hybrid retrieval answer directly for design parameters")
                
                parts = rag_context.split("[SUPPORTING DOCUMENT EXCERPTS]")
                draft_answer = parts[0].replace("[HYBRID RETRIEVAL ANALYSIS]", "").strip()
                
                state["agent_response"] = {
                    "hybrid_answer_used": True,
                    "draft_answer": draft_answer,
                    "rag_context": rag_context
                }
                return state
            
            if not document_text:
                if rag_context:
                    logger.info("No full document text; answering from RAG context")
                    answer = await self.llm.chat_completion(
                        messages=[
                            {"role": "system", "content": GENERAL_QUESTION_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": f"RETRIEVED TECHNICAL DOCUMENT CONTEXT:\n{rag_context}\n\nUser Question: {state.get('user_query', '')}\n\nExtract design criteria, calculations, standards compliance, and engineering requirements.",
                            },
                        ],
                        temperature=0.5,
                        max_tokens=2000,
                    )
                    state["agent_response"] = {"answer": answer, "from_rag_context": True}
                    return state
                state["error"] = "No document text available for design parameters"
                state["agent_response"] = {}
                return state
            
            # Use RAG context for design parameters
            if rag_context:
                logger.info("Using RAG context for design parameters extraction")
                answer = await self.llm.chat_completion(
                    messages=[
                        {"role": "system", "content": GENERAL_QUESTION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"TECHNICAL DOCUMENT CONTEXT:\n{rag_context}\n\nUser Question: {state.get('user_query', '')}\n\nExtract and present design parameters, calculations, standards (ISO, ASME, ANSI), compliance requirements, and engineering criteria with values and units.",
                        },
                    ],
                    temperature=0.5,
                    max_tokens=2000,
                )
                state["agent_response"] = {"answer": answer, "from_rag_context": True}
            else:
                state["agent_response"] = {"answer": "No design parameters found in the document."}
            
            logger.info("Design parameters extraction completed")
        except Exception as e:
            logger.error(f"Design parameters error: {e}")
            state["error"] = str(e)
            state["agent_response"] = {}
        return state
    
    async def _handle_general_question(self, state: ConversationState) -> ConversationState:
        """Handle general questions: meta chat (CandexAI assistant) or document Q&A."""
        try:
            document_text = state.get("document_text", "")
            user_query = state.get("user_query", "")
            rag_context = state.get("rag_context", "")
            history = state.get("conversation_history") or []
            retrieval_meta = state.get("retrieval_metadata") or {}
            collection_name = retrieval_meta.get("collection_name", "")

            document_available = bool(
                document_text
                and "upload a document first" not in document_text.lower()[:200]
            )

            if is_meta_conversational_query(user_query):
                answer = await self.generate_conversational_response(
                    user_query=user_query,
                    conversation_history=history,
                    collection_name=collection_name,
                    document_available=document_available,
                )
                state["agent_response"] = {"answer": answer, "conversational": True}
                return state

            if not document_text or not document_available:
                answer = await self.generate_conversational_response(
                    user_query=user_query,
                    conversation_history=history,
                    collection_name=collection_name,
                    document_available=False,
                )
                state["agent_response"] = {"answer": answer, "conversational": True}
                return state

            rag_section = ""
            if rag_context:
                rag_section = f"\n\nRELEVANT CONTEXT FROM KNOWLEDGE BASE:\n{rag_context[:6000]}"

            user_prompt = f"""Document Content:
{document_text[:8000]}
{rag_section}

User Question: {user_query}

Please provide a helpful answer based on the document and any relevant context."""

            answer = await self.conversational_llm.chat_completion(
                messages=[
                    {"role": "system", "content": GENERAL_QUESTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=2000,
            )
            if not (answer and answer.strip()):
                answer = await self.generate_conversational_response(
                    user_query=user_query,
                    conversation_history=history,
                    collection_name=collection_name,
                    document_available=True,
                )
            state["agent_response"] = {"answer": answer}

        except Exception as e:
            logger.error(f"General question error: {e}")
            state["error"] = str(e)
            state["agent_response"] = {"answer": CANDEXAI_ASSISTANT_FALLBACK}
        return state
    
    async def _format_response(self, state: ConversationState) -> ConversationState:
        """Format the agent response using LLM for natural, context-aware formatting."""
        intent = state.get("intent", "")
        agent_response = state.get("agent_response", {})
        error = state.get("error")
        user_query = state.get("user_query", "")
        
        if error:
            state["final_response"] = f"I encountered an error: {error}"
            return state

        # Conversational / general answers are already final prose
        if intent == IntentType.GENERAL_QUESTION:
            answer = (agent_response.get("answer") or "").strip()
            if answer:
                state["final_response"] = answer
                return state

        # Hybrid path: draft from generate_answer is already counsel-ready — skip second LLM (~40s)
        if agent_response.get("hybrid_answer_used"):
            draft = (agent_response.get("draft_answer") or "").strip()
            if len(draft) >= 80:
                logger.info(
                    "⚡ SKIP format pass: using hybrid draft as final response "
                    f"({len(draft)} chars)"
                )
                state["final_response"] = draft
                return state
        
        # Use LLM to format the response based on user query and agent output
        try:
            formatted = await self._llm_format_response(
                user_query=user_query,
                intent=intent,
                agent_data=agent_response
            )
            state["final_response"] = formatted
            return state
        except Exception as e:
            logger.warning(f"LLM formatting failed, using fallback: {e}")
            # Fallback to simple formatting if LLM fails
            state["final_response"] = self._fallback_format(intent, agent_response)
            return state
    
    async def _llm_format_response(
        self,
        user_query: str,
        intent: str,
        agent_data: Dict[str, Any]
    ) -> str:
        """Format the agent response into a natural, user-friendly message."""

        if agent_data.get("hybrid_answer_used"):
            system_prompt = HYBRID_ANSWER_FORMAT_PROMPT
            rag_context = agent_data.get("rag_context", "")
            draft_answer = agent_data.get("draft_answer", "")
            if not draft_answer and "[HYBRID RETRIEVAL ANALYSIS]" in rag_context:
                parts = rag_context.split("[SUPPORTING DOCUMENT EXCERPTS]")
                draft_answer = parts[0].replace("[HYBRID RETRIEVAL ANALYSIS]", "").strip()
            agent_data = {"draft_answer": draft_answer, **{k: v for k, v in agent_data.items() if k != "rag_context"}}
            format_temperature = 0.62
        elif intent == IntentType.TECHNICAL_SPECS:
            system_prompt = CLAUSE_EXTRACTION_FORMAT_PROMPT
            format_temperature = 0.5
        elif intent == IntentType.SAFETY_INFO:
            system_prompt = RISK_ANALYSIS_FORMAT_PROMPT
            format_temperature = 0.5
        elif intent == IntentType.DESIGN_PARAMETERS:
            system_prompt = COMPARISON_FORMAT_PROMPT
            format_temperature = 0.5
        elif intent == IntentType.TROUBLESHOOTING:
            system_prompt = REPORT_GENERATION_FORMAT_PROMPT
            format_temperature = 0.5
        else:
            system_prompt = GENERAL_FORMAT_PROMPT
            format_temperature = 0.5

        agent_data_str = json.dumps(agent_data, indent=2, default=str)
        
        # Truncate if too long
        if len(agent_data_str) > 15000:
            agent_data_str = agent_data_str[:15000] + "\n... [truncated for length]"
        
        if agent_data.get("hybrid_answer_used"):
            user_prompt = f"""Engineer's question: "{user_query}"

Draft analysis and structured metadata (JSON):
{agent_data_str}

Deliver a technical advisory in senior-engineer prose. Use document_clause_count (or total_clauses) for any section-count statement. Keep every citation accurate. Include units with all numerical values. No bullet templates or chatbot section headers."""
        else:
            user_prompt = f"""Engineer's question: "{user_query}"

Analysis type: {intent}

Analysis results (JSON):
{agent_data_str}

Deliver a technical advisory that directly answers the question. Use engineering report-style paragraphs with units for numerical values, not robotic templates or bullet dumps unless a list was requested."""

        formatted = await self.llm.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=format_temperature,
            max_tokens=4000,
        )
        logger.info(f"LLM formatted response for intent: {intent}")
        return formatted
    
    def _fallback_format(self, intent: str, agent_response: Dict[str, Any]) -> str:
        """Simple fallback formatting if LLM formatting fails."""
        if intent == IntentType.TECHNICAL_SPECS:
            total = agent_response.get("total_clauses", 0)
            return f"Found {total} technical sections in the document. See 'data' field for details."
        elif intent == IntentType.SAFETY_INFO:
            total = agent_response.get("total_risks", 0)
            critical = agent_response.get("critical_risks", 0)
            return f"Found {total} safety items ({critical} critical). See 'data' field for details."
        elif intent == IntentType.DESIGN_PARAMETERS:
            return "Design parameters extracted. See 'data' field for details."
        elif intent == IntentType.TROUBLESHOOTING:
            return "Troubleshooting guidance provided. See 'data' field for details."
        else:
            return agent_response.get("answer", "Analysis complete. See 'data' field for details.")
    
    async def chat(
        self,
        user_query: str,
        document_text: str,
        rag_context: str = None,
        session_id: str = None,
        conversation_history: List[Dict[str, str]] = None,
        clause_index: Dict[str, Dict[str, Any]] = None,
        retrieval_metadata: Optional[Dict[str, Any]] = None,
        collection_name: str = "",
        document_name: str = "",
    ) -> Dict[str, Any]:
        """
        Process a natural language query about the technical document.
        
        Args:
            user_query: User's natural language question
            document_text: The technical document text to analyze
            rag_context: Optional RAG context from knowledge base
            session_id: Session ID for checkpointing (enables conversation memory)
            conversation_history: Previous conversation messages for context
            clause_index: Section index from hybrid retrieval for citation details
            
        Returns:
            Dict with response, intent, and raw agent data
        """
        # Store clause_index for building citations
        self._current_clause_index = clause_index or {}
        # Sanitize conversation_history to remove MongoDB ObjectId (not serializable)
        sanitized_history = []
        if conversation_history:
            for msg in conversation_history:
                sanitized_msg = {}
                for key, value in msg.items():
                    # Skip ObjectId fields or convert to string
                    if key == "_id" or hasattr(value, '__str__') and 'ObjectId' in str(type(value)):
                        sanitized_msg[key] = str(value)
                    else:
                        sanitized_msg[key] = value
                sanitized_history.append(sanitized_msg)
        
        initial_state: ConversationState = {
            "user_query": user_query,
            "document_text": document_text,
            "rag_context": rag_context,
            "intent": "",
            "agent_response": {},
            "final_response": "",
            "error": None,
            "session_id": session_id,
            "conversation_history": sanitized_history,
            "retrieval_metadata": {
                **(retrieval_metadata or {}),
                "collection_name": collection_name or (retrieval_metadata or {}).get("collection_name", ""),
            },
        }
        
        try:
            # Use thread_id for checkpointing if session_id provided
            config = {}
            if session_id and self._checkpointer:
                config = {"configurable": {"thread_id": session_id}}
                logger.info(f"Using MongoDB checkpointer with thread_id: {session_id}")
            
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            # Build citation details from clause_index
            agent_response = final_state.get("agent_response", {})
            if self._current_clause_index and agent_response.get("clause_ids"):
                citations = {}
                for clause_id in agent_response.get("clause_ids", []):
                    if clause_id in self._current_clause_index:
                        clause_info = self._current_clause_index[clause_id]
                        citations[clause_id] = {
                            "clause_type": clause_info.get("clause_type", "Unknown"),
                            "content": clause_info.get("content", ""),
                            "importance": clause_info.get("importance", "medium"),
                            "source_chunk_id": clause_info.get("source_chunk_id", ""),
                            "parent_clause_id": clause_info.get("parent_clause_id")
                        }
                agent_response["citations"] = citations
                logger.info(f"📚 Built {len(citations)} citation details from clause_index")
            
            return {
                "status": "success",
                "query": user_query,
                "intent": final_state.get("intent", "unknown"),
                "response": final_state.get("final_response", ""),
                "data": agent_response,
                "error": final_state.get("error"),
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"Conversation graph error: {e}")
            return {
                "status": "error",
                "query": user_query,
                "intent": "unknown",
                "response": f"An error occurred: {str(e)}",
                "data": {},
                "error": str(e),
                "session_id": session_id
            }


# Singleton instance
conversation_graph = ConversationGraph()
