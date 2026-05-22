"""
MongoDB Service for storing chat history and session instances.
Uses Motor for async operations and PyMongo for sync operations.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from utils.logger import get_logger

logger = get_logger("MongoDBService")


class MongoDBService:
    """
    MongoDB service for managing chat history and session instances.
    Provides both sync and async operations.
    """
    
    _instance = None
    _async_client = None
    _sync_client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.uri = config.MONGO_DB_URI
        self.db_name = config.MONGO_DB_NAME
        self.chat_history_collection = config.MONGO_CHAT_HISTORY_COLLECTION
        self.instances_collection = config.MONGO_INSTANCES_COLLECTION
        
        # Initialize clients
        self._init_clients()
        self._initialized = True
        logger.info(f"MongoDBService initialized with database: {self.db_name}")
    
    def _init_clients(self):
        """Initialize MongoDB clients."""
        try:
            # Async client for async operations
            self._async_client = AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                maxPoolSize=200,  # Increased for load testing
                minPoolSize=10
            )
            self._async_db = self._async_client[self.db_name]
            
            # Sync client for sync operations
            self._sync_client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                maxPoolSize=200,  # Increased for load testing
                minPoolSize=10
            )
            self._sync_db = self._sync_client[self.db_name]
            
            # Test connection
            self._sync_client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    @property
    def async_chat_history(self):
        """Get async chat history collection."""
        return self._async_db[self.chat_history_collection]
    
    @property
    def async_instances(self):
        """Get async instances collection."""
        return self._async_db[self.instances_collection]
    
    @property
    def sync_chat_history(self):
        """Get sync chat history collection."""
        return self._sync_db[self.chat_history_collection]
    
    @property
    def sync_instances(self):
        """Get sync instances collection."""
        return self._sync_db[self.instances_collection]
    
    # ==================== Instance Operations ====================
    
    async def create_instance(
        self,
        session_id: str,
        document_name: str,
        document_text: str,
        template_name: Optional[str] = None,
        template_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new session instance in MongoDB.
        
        Args:
            session_id: Unique session identifier
            document_name: Name of the uploaded document
            document_text: Extracted text from the document
            template_name: Optional template document name
            template_text: Optional template document text
            metadata: Additional metadata
            
        Returns:
            Created instance document
        """
        instance = {
            "session_id": session_id,
            "document_name": document_name,
            "document_text": document_text,
            "template_name": template_name,
            "template_text": template_text,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "status": "active"
        }
        
        # Upsert to handle re-uploads
        result = await self.async_instances.update_one(
            {"session_id": session_id},
            {"$set": instance},
            upsert=True
        )
        
        logger.info(f"Instance created/updated for session: {session_id}")
        return instance
    
    async def get_instance(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get instance by session_id."""
        instance = await self.async_instances.find_one({"session_id": session_id})
        return instance

    async def get_instance_by_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent instance for a Qdrant collection name."""
        instance = await self.async_instances.find_one(
            {"metadata.collection_name": collection_name},
            sort=[("created_at", -1)],
        )
        return instance
    
    async def update_instance(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an existing instance."""
        updates["updated_at"] = datetime.utcnow()
        result = await self.async_instances.update_one(
            {"session_id": session_id},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    async def delete_instance(self, session_id: str) -> bool:
        """Delete an instance and its chat history."""
        # Delete instance
        result = await self.async_instances.delete_one({"session_id": session_id})
        
        # Delete associated chat history
        await self.async_chat_history.delete_many({"session_id": session_id})
        
        logger.info(f"Instance and chat history deleted for session: {session_id}")
        return result.deleted_count > 0
    
    async def list_instances(
        self,
        limit: int = 50,
        skip: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all instances with pagination."""
        query = {}
        if status:
            query["status"] = status
            
        cursor = self.async_instances.find(
            query,
            {"document_text": 0, "template_text": 0}  # Exclude large text fields
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        return await cursor.to_list(length=limit)

    # ==================== Clause Index Storage (large documents) ====================

    @property
    def async_clause_indices(self):
        """Collection for per-document clause indices."""
        return self._async_db[config.MONGO_CLAUSE_INDEX_COLLECTION]

    async def save_clause_index(
        self,
        collection_name: str,
        clause_index: Dict[str, Dict[str, Any]],
        clause_type_counts: Optional[Dict[str, int]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Persist clause index for collections too large to pass to LLM clause selection."""
        doc = {
            "collection_name": collection_name,
            "clause_index": clause_index,
            "clause_type_counts": clause_type_counts or {},
            "clause_count": len(clause_index),
            "session_id": session_id,
            "updated_at": datetime.utcnow(),
        }
        await self.async_clause_indices.update_one(
            {"collection_name": collection_name},
            {"$set": doc},
            upsert=True,
        )
        logger.info(
            f"Clause index saved to MongoDB for '{collection_name}' "
            f"({len(clause_index)} clauses)"
        )

    async def get_stored_clause_index(
        self, collection_name: str
    ) -> Optional[Dict[str, Any]]:
        """Load persisted clause index by Qdrant collection name."""
        return await self.async_clause_indices.find_one(
            {"collection_name": collection_name}
        )
    
    # ==================== Chat History Operations ====================
    
    async def add_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        agent_data: Optional[Dict[str, Any]] = None,
        rag_used: bool = False,
        rag_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a chat message to history.
        
        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            intent: Detected intent (for assistant messages)
            agent_data: Raw agent response data
            rag_used: Whether RAG was used
            rag_context: RAG context if used
            metadata: Additional metadata
            
        Returns:
            Created message document
        """
        message = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "intent": intent,
            "agent_data": agent_data,
            "rag_used": rag_used,
            "rag_context": rag_context[:1000] if rag_context else None,  # Truncate RAG context
            "metadata": metadata or {},
            "timestamp": datetime.utcnow()
        }
        
        result = await self.async_chat_history.insert_one(message)
        message["_id"] = result.inserted_id
        
        logger.debug(f"Chat message added for session: {session_id}, role: {role}")
        return message
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 50,
        include_agent_data: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum messages to return
            include_agent_data: Whether to include raw agent data
            
        Returns:
            List of chat messages
        """
        projection = {
            "session_id": 1,
            "role": 1,
            "content": 1,
            "intent": 1,
            "rag_used": 1,
            "timestamp": 1
        }
        
        if include_agent_data:
            projection["agent_data"] = 1
            projection["rag_context"] = 1
        
        cursor = self.async_chat_history.find(
            {"session_id": session_id},
            projection
        ).sort("timestamp", 1).limit(limit)
        
        return await cursor.to_list(length=limit)
    
    async def get_recent_messages(
        self,
        session_id: str,
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get the most recent messages for context."""
        cursor = self.async_chat_history.find(
            {"session_id": session_id},
            {"role": 1, "content": 1, "intent": 1, "timestamp": 1}
        ).sort("timestamp", -1).limit(count)
        
        messages = await cursor.to_list(length=count)
        return list(reversed(messages))  # Return in chronological order
    
    async def clear_chat_history(self, session_id: str) -> int:
        """Clear all chat history for a session."""
        result = await self.async_chat_history.delete_many({"session_id": session_id})
        logger.info(f"Cleared {result.deleted_count} messages for session: {session_id}")
        return result.deleted_count
    
    # ==================== Conversation Context ====================
    
    async def get_conversation_context(
        self,
        session_id: str,
        max_messages: int = 10,
        max_chars: int = 4000
    ) -> str:
        """
        Get formatted conversation context for LLM.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum messages to include
            max_chars: Maximum characters for context
            
        Returns:
            Formatted conversation history string
        """
        messages = await self.get_recent_messages(session_id, max_messages)
        
        if not messages:
            return ""
        
        context_parts = []
        total_chars = 0
        
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            
            # Truncate individual messages if needed
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            line = f"{role}: {content}"
            
            if total_chars + len(line) > max_chars:
                break
                
            context_parts.append(line)
            total_chars += len(line)
        
        return "\n\n".join(context_parts)
    
    # ==================== Utility Methods ====================
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        instance = await self.get_instance(session_id)
        message_count = await self.async_chat_history.count_documents({"session_id": session_id})
        
        return {
            "session_id": session_id,
            "exists": instance is not None,
            "document_name": instance.get("document_name") if instance else None,
            "has_template": instance.get("template_text") is not None if instance else False,
            "message_count": message_count,
            "created_at": instance.get("created_at") if instance else None,
            "updated_at": instance.get("updated_at") if instance else None
        }
    
    def close(self):
        """Close MongoDB connections."""
        if self._async_client:
            self._async_client.close()
        if self._sync_client:
            self._sync_client.close()
        logger.info("MongoDB connections closed")


# Singleton instance
mongodb_service = None

def get_mongodb_service() -> MongoDBService:
    """Get or create MongoDB service singleton."""
    global mongodb_service
    if mongodb_service is None:
        mongodb_service = MongoDBService()
    return mongodb_service
