#!/usr/bin/env python3
"""
Test script to ingest a mechanical engineering document and run test queries.
"""
import requests
import os
import time

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"
PDF_PATH = "Manual Thermal Power Plant-CandexAI-ingest.pdf"
COLLECTION_NAME = "me-test1"
SESSION_ID = "test_session_001"

def upload_document():
    """Upload the thermal power plant manual."""
    print(f"\n{'='*60}")
    print("📤 UPLOADING DOCUMENT")
    print(f"{'='*60}")
    
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: File not found: {PDF_PATH}")
        print(f"   Current directory: {os.getcwd()}")
        return False
    
    print(f"📄 File: {PDF_PATH}")
    print(f"📦 Collection: {COLLECTION_NAME}")
    
    with open(PDF_PATH, 'rb') as f:
        files = {'document': (PDF_PATH, f, 'application/pdf')}
        data = {'collection_name': COLLECTION_NAME}
        
        print("\n⏳ Uploading and processing (this may take a few minutes)...")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/upload-document",
                files=files,
                data=data,
                timeout=600  # 10 minute timeout for large documents
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Upload successful! (took {elapsed:.1f}s)")
                print(f"   Collection: {result.get('collection_name')}")
                print(f"   Chunks: {result.get('total_chunks', 'N/A')}")
                print(f"   Pages: {result.get('total_pages', 'N/A')}")
                return True
            else:
                print(f"\n❌ Upload failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"\n⚠️  Upload timed out after {elapsed:.1f}s")
            print("   The document may still be processing. Check /collections endpoint.")
            return False
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False

def ask_question(question, description=""):
    """Ask a question about the uploaded document."""
    print(f"\n{'='*60}")
    print(f"❓ QUERY: {question}")
    if description:
        print(f"   ({description})")
    print(f"{'='*60}")
    
    data = {
        'session_id': SESSION_ID,
        'message': question,
        'collection_name': COLLECTION_NAME,
        'use_rag': 'true',
        'hybrid_mode': 'true'
    }
    
    print("⏳ Processing query...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            data=data,
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Response received (took {elapsed:.1f}s)")
            print(f"\n📝 ANSWER:")
            print("-" * 60)
            print(result.get('response', 'No response'))
            print("-" * 60)
            
            # Show metadata
            if 'metadata' in result:
                meta = result['metadata']
                print(f"\n📊 Metadata:")
                print(f"   Intent: {meta.get('intent', 'N/A')}")
                print(f"   RAG Used: {meta.get('rag_used', 'N/A')}")
                print(f"   Chunks Retrieved: {meta.get('chunks_retrieved', 'N/A')}")
                print(f"   Total Latency: {meta.get('total_latency_seconds', 'N/A')}s")
            
            return True
        else:
            print(f"\n❌ Query failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    """Main test flow."""
    print("\n" + "="*60)
    print("🔧 MECHANICAL ENGINEERING CHATBOT - INGESTION TEST")
    print("="*60)
    
    # Step 1: Upload document
    if not upload_document():
        print("\n❌ Upload failed. Exiting.")
        return
    
    print("\n⏸️  Waiting 3 seconds before querying...")
    time.sleep(3)
    
    # Step 2: Run test queries
    test_queries = [
        {
            "question": "What are the main components of a thermal power plant?",
            "description": "General technical specs"
        },
        {
            "question": "What are the safety precautions mentioned in the manual?",
            "description": "Safety information"
        },
        {
            "question": "What is the operating temperature range for the boiler?",
            "description": "Technical specifications with units"
        },
        {
            "question": "How do I troubleshoot low steam pressure?",
            "description": "Troubleshooting guidance"
        },
        {
            "question": "What design standards does this plant follow?",
            "description": "Design parameters and standards"
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*60}")
        print(f"# TEST QUERY {i}/{len(test_queries)}")
        print(f"{'#'*60}")
        
        ask_question(query["question"], query["description"])
        
        if i < len(test_queries):
            print("\n⏸️  Waiting 2 seconds before next query...")
            time.sleep(2)
    
    # Summary
    print("\n\n" + "="*60)
    print("✅ TESTING COMPLETE")
    print("="*60)
    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Session: {SESSION_ID}")
    print(f"\nYou can now:")
    print(f"  - View session history: GET {API_BASE_URL}/session/{SESSION_ID}/history")
    print(f"  - Ask more questions using the /chat endpoint")
    print(f"  - Check collections: GET {API_BASE_URL}/collections")

if __name__ == "__main__":
    main()
