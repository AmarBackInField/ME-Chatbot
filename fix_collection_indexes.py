#!/usr/bin/env python3
"""
Script to add missing payload indexes to existing Qdrant collections.
Run this to fix collections that were created before the index update.
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv()

def fix_collection_indexes(collection_name: str):
    """Add missing payload indexes to an existing collection."""
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        return False
    
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # Check if collection exists
    try:
        collections = client.get_collections().collections
        if not any(col.name == collection_name for col in collections):
            print(f"Error: Collection '{collection_name}' does not exist")
            return False
    except Exception as e:
        print(f"Error checking collections: {e}")
        return False
    
    print(f"Adding payload indexes to collection '{collection_name}'...")
    
    # Indexes to create
    indexes = [
        ("source_collection", PayloadSchemaType.KEYWORD),
        ("document_id", PayloadSchemaType.KEYWORD),
        ("chunk_id", PayloadSchemaType.KEYWORD),
        ("layer", PayloadSchemaType.INTEGER)
    ]
    
    success_count = 0
    for field_name, field_schema in indexes:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema
            )
            print(f"✅ Created payload index on '{field_name}'")
            success_count += 1
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"ℹ️  Payload index on '{field_name}' already exists")
                success_count += 1
            else:
                print(f"❌ Failed to create index on '{field_name}': {e}")
    
    print(f"\n✅ Successfully created/verified {success_count}/{len(indexes)} indexes")
    return success_count == len(indexes)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
    else:
        collection_name = "amar6"  # Default collection
    
    print(f"Fixing indexes for collection: {collection_name}\n")
    success = fix_collection_indexes(collection_name)
    
    if success:
        print(f"\n🎉 Collection '{collection_name}' is now ready to use!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some indexes could not be created. Check the errors above.")
        sys.exit(1)
