import os
import sys
import asyncio
import tiktoken
import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Callable, Coroutine
from dataclasses import dataclass, field, asdict
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import config
from llm_service import LLMService
from utils.logger import get_logger
from prompts import (
    get_clause_extraction_system_prompt,
    get_l2_summary_prompt,
    get_short_summary_prompt,
    SHORT_SUMMARY_SYSTEM_PROMPT,
    L2_PROCESSING_SYSTEM_PROMPT
)

logger = get_logger("ChunkingService")


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail fast
    - HALF_OPEN: Testing if service recovered
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
        logger.info(f"🔌 CircuitBreaker initialized (threshold={failure_threshold}, "
                   f"recovery={recovery_timeout}s)")
    
    @property
    def state(self) -> str:
        return self._state
    
    async def can_execute(self) -> bool:
        """Check if request can proceed based on circuit state."""
        async with self._lock:
            if self._state == self.CLOSED:
                return True
            
            if self._state == self.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("🔄 Circuit breaker: OPEN -> HALF_OPEN (testing recovery)")
                    return True
                return False
            
            if self._state == self.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            return False
    
    async def record_success(self):
        """Record a successful request."""
        async with self._lock:
            self._success_count += 1
            
            if self._state == self.HALF_OPEN:
                if self._success_count >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("✅ Circuit breaker: HALF_OPEN -> CLOSED (service recovered)")
    
    async def record_failure(self):
        """Record a failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                logger.warning("⚠️ Circuit breaker: HALF_OPEN -> OPEN (recovery failed)")
            elif self._state == self.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(f"🔴 Circuit breaker: CLOSED -> OPEN "
                                 f"(failures={self._failure_count}/{self.failure_threshold})")
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0


class LLMRequestQueue:
    """
    Queue-based parallel processing for LLM API calls.
    
    Features:
    - Configurable max concurrent workers (default: 10)
    - Request ID tracking for result retrieval
    - Automatic retry with exponential backoff
    - Circuit breaker pattern to prevent cascading failures
    - Adaptive rate limiting based on error rate
    - Progress tracking and logging
    
    This provides ~4x speedup over sequential batch processing.
    """
    
    def __init__(
        self, 
        llm_service: 'LLMService',
        max_workers: int = None,
        max_retries: int = None,
        retry_base_delay: float = None,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_recovery: float = 30.0,
        adaptive_rate_enabled: bool = True
    ):
        self.llm_service = llm_service
        self.max_workers = max_workers or config.LLM_MAX_CONCURRENT
        self.max_retries = max_retries or config.LLM_RETRY_MAX
        self.retry_base_delay = retry_base_delay or config.LLM_RETRY_BASE_DELAY
        
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}
        self._completed: int = 0
        self._total: int = 0
        self._start_time: float = 0
        
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=circuit_breaker_recovery
        )
        
        self._adaptive_rate_enabled = adaptive_rate_enabled
        self._current_workers = self.max_workers
        self._error_window: List[float] = []
        self._error_window_size = 20
        self._error_rate_threshold = 0.3
        self._min_workers = 2
        
        logger.info(f"🚀 LLMRequestQueue initialized (max_workers={self.max_workers}, "
                   f"max_retries={self.max_retries}, retry_delay={self.retry_base_delay}s)")
        logger.info(f"   Circuit breaker: threshold={circuit_breaker_threshold}, "
                   f"recovery={circuit_breaker_recovery}s")
        logger.info(f"   Adaptive rate limiting: {'enabled' if adaptive_rate_enabled else 'disabled'}")
    
    async def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_workers)
        return self._semaphore
    
    async def _update_error_rate(self, is_error: bool):
        """Update error rate tracking for adaptive rate limiting."""
        if not self._adaptive_rate_enabled:
            return
        
        self._error_window.append(1.0 if is_error else 0.0)
        
        if len(self._error_window) > self._error_window_size:
            self._error_window.pop(0)
        
        if len(self._error_window) >= 5:
            error_rate = sum(self._error_window) / len(self._error_window)
            
            if error_rate > self._error_rate_threshold:
                new_workers = max(self._min_workers, self._current_workers - 2)
                if new_workers < self._current_workers:
                    logger.warning(f"📉 Adaptive rate: reducing workers {self._current_workers} -> {new_workers} "
                                 f"(error_rate={error_rate:.1%})")
                    self._current_workers = new_workers
                    self._semaphore = asyncio.Semaphore(new_workers)
            elif error_rate < 0.1 and self._current_workers < self.max_workers:
                new_workers = min(self.max_workers, self._current_workers + 1)
                if new_workers > self._current_workers:
                    logger.info(f"📈 Adaptive rate: increasing workers {self._current_workers} -> {new_workers} "
                              f"(error_rate={error_rate:.1%})")
                    self._current_workers = new_workers
                    self._semaphore = asyncio.Semaphore(new_workers)
    
    def _count_message_tokens(self, messages: List[Dict[str, str]]) -> Tuple[int, int]:
        """Count tokens in messages. Returns (total_tokens, total_chars)."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            total_tokens = 0
            total_chars = 0
            for msg in messages:
                content = msg.get("content", "")
                total_tokens += len(encoding.encode(content))
                total_chars += len(content)
            return total_tokens, total_chars
        except Exception:
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            return total_chars // 4, total_chars  # Rough estimate
    
    async def _execute_with_retry(
        self,
        request_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4000
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Execute a single LLM request with retry logic, circuit breaker, and adaptive rate limiting.
        
        Args:
            request_id: Unique identifier for this request
            messages: Chat messages for LLM
            temperature: LLM temperature
            max_tokens: Max tokens in response
        
        Returns:
            Tuple of (request_id, response_text, error_message)
        """
        prompt_tokens, prompt_chars = self._count_message_tokens(messages)
        
        if not await self._circuit_breaker.can_execute():
            self._completed += 1
            logger.warning(f"🔴 Request {request_id} rejected by circuit breaker (state={self._circuit_breaker.state})")
            logger.warning(f"   📊 Request details: {prompt_tokens} tokens, {prompt_chars} chars")
            return (request_id, None, "Circuit breaker OPEN - service unavailable")
        
        semaphore = await self._get_semaphore()
        
        async with semaphore:
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    if attempt > 0:
                        wait_time = self.retry_base_delay * (2 ** (attempt - 1))
                        logger.info(f"   🔄 Retry {attempt}/{self.max_retries-1} for {request_id} after {wait_time:.1f}s")
                        logger.info(f"      📊 Retrying: {prompt_tokens} tokens, {prompt_chars} chars")
                        await asyncio.sleep(wait_time)
                    
                    response = await self.llm_service.chat_completion(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    if response and not response.strip().startswith('<!DOCTYPE') and not response.strip().startswith('<html'):
                        self._completed += 1
                        await self._circuit_breaker.record_success()
                        await self._update_error_rate(is_error=False)
                        return (request_id, response, None)
                    else:
                        # #region agent log
                        try:
                            import json as _json, time as _time
                            with open("/Users/amarchoudhary/Desktop/Legal_CandexAI/.cursor/debug-e24b7b.log", "a", encoding="utf-8") as _df:
                                _json.dump({
                                    "sessionId": "e24b7b",
                                    "hypothesisId": "A",
                                    "location": "chunking.py:_execute_with_retry:empty_response",
                                    "message": "Treated as 504 empty/HTML",
                                    "data": {
                                        "request_id": request_id,
                                        "response_is_none": response is None,
                                        "response_len": len(response) if response else 0,
                                        "response_repr": repr(response)[:200] if response else "",
                                        "attempt": attempt + 1,
                                    },
                                    "timestamp": int(_time.time() * 1000),
                                }, _df)
                                _df.write("\n")
                        except Exception:
                            pass
                        # #endregion
                        last_error = "Empty or HTML error response (504 Gateway Timeout)"
                        await self._circuit_breaker.record_failure()
                        await self._update_error_rate(is_error=True)
                        
                        # Log detailed info for 504 errors
                        logger.error(f"🚨 504 ERROR for {request_id} (attempt {attempt+1}/{self.max_retries})")
                        logger.error(f"   📊 Prompt: {prompt_tokens} tokens | {prompt_chars} chars")
                        logger.error(f"   📊 Max output tokens requested: {max_tokens}")
                        logger.error(f"   📊 Temperature: {temperature}")
                        logger.error(f"   📊 Circuit breaker: {self._circuit_breaker.state} | Workers: {self._current_workers}")
                        if messages:
                            user_content = messages[-1].get("content", "")[:200]
                            logger.error(f"   📝 Prompt preview: {user_content}...")
                        
                except Exception as e:
                    last_error = str(e)
                    await self._circuit_breaker.record_failure()
                    await self._update_error_rate(is_error=True)
                    
                    # Log detailed info for exceptions
                    logger.error(f"🚨 EXCEPTION for {request_id} (attempt {attempt+1}/{self.max_retries})")
                    logger.error(f"   📊 Prompt: {prompt_tokens} tokens | {prompt_chars} chars")
                    logger.error(f"   📊 Max output tokens requested: {max_tokens}")
                    logger.error(f"   📊 Circuit breaker: {self._circuit_breaker.state} | Workers: {self._current_workers}")
                    logger.error(f"   ❌ Error: {str(e)[:200]}")
            
            self._completed += 1
            logger.warning(f"❌ Request {request_id} FAILED after {self.max_retries} retries")
            logger.warning(f"   📊 Final stats: {prompt_tokens} tokens, {prompt_chars} chars")
            logger.warning(f"   📊 Last error: {last_error[:100] if last_error else 'Unknown'}")
            return (request_id, None, last_error)
    
    async def process_batch(
        self,
        requests: List[Tuple[str, List[Dict[str, str]]]],
        temperature: float = 0.1,
        max_tokens: int = 4000,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        """
        Process a batch of LLM requests in parallel using queue-based workers.
        
        Args:
            requests: List of (request_id, messages) tuples
            temperature: LLM temperature for all requests
            max_tokens: Max tokens for all requests
            progress_callback: Optional callback(completed, total, rate) for progress updates
        
        Returns:
            Dict mapping request_id -> (response_text, error_message)
        """
        self._results = {}
        self._errors = {}
        self._completed = 0
        self._total = len(requests)
        self._start_time = time.time()
        
        logger.info(f"🚀 Starting parallel processing of {self._total} requests (max_concurrent={self.max_workers})")
        
        tasks = [
            self._execute_with_retry(req_id, messages, temperature, max_tokens)
            for req_id, messages in requests
        ]
        
        results = {}
        success_count = 0
        fail_count = 0
        
        pbar = tqdm(
            total=self._total,
            desc="🔄 L2 Processing",
            unit="chunk",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] ✅{postfix}",
            colour="green"
        )
        
        for coro in asyncio.as_completed(tasks):
            request_id, response, error = await coro
            results[request_id] = (response, error)
            
            if response is not None:
                success_count += 1
            else:
                fail_count += 1
            
            pbar.update(1)
            pbar.set_postfix_str(f"{success_count} ❌{fail_count} | CB:{self._circuit_breaker.state[:1].upper()} W:{self._current_workers}")
            
            if progress_callback:
                elapsed = time.time() - self._start_time
                rate = self._completed / elapsed if elapsed > 0 else 0
                progress_callback(self._completed, self._total, rate)
        
        pbar.close()
        
        elapsed = time.time() - self._start_time
        logger.info(f"✅ Batch complete: {success_count}/{self._total} successful in {elapsed:.1f}s "
                   f"({self._total/elapsed:.1f} req/sec)")
        
        return results


@dataclass
class Chunk:
    """Represents a chunk at any layer in the hierarchy."""
    layer: int
    chunk_id: str
    text: str
    token_count: int
    clauses: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: str = "no_risk"
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    clause_relationships: List[Dict[str, Any]] = field(default_factory=list)
    short_summary: Optional[str] = None  # Short summary for L2+ chunks (size from config.SUMMARY_TARGET_SIZE)
    clause_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Quick lookup: clause_id -> {source, type, parent}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for tree representation."""
        return {
            "chunk_id": self.chunk_id,
            "layer": self.layer,
            "token_count": self.token_count,
            "risk_level": self.risk_level,
            "clauses": self.clauses,
            "clause_relationships": self.clause_relationships,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "metadata": self.metadata,
            "short_summary": self.short_summary,
            "clause_index": self.clause_index
        }


class HierarchicalChunker:
    """
    Hierarchical chunking strategy for legal documents.
    
    Creates multiple layers:
    - Layer 1 (L1): Original document split into chunks of LAYER1_CHUNK_SIZE tokens
    - Layer 2 (L2): Summaries of L1 chunks with clause extraction and risk analysis
    - Layer 3+ (L3+): Progressive summarization when accumulated tokens exceed threshold
    """
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.layer_chunk_size = config.LAYER_CHUNK_SIZE
        self.summary_target_size = config.SUMMARY_TARGET_SIZE
        self.layer_threshold = config.LAYER_THRESHOLD
        self.batch_size = config.CHUNK_BATCH_SIZE
        
        self.queue_enabled = config.LLM_QUEUE_ENABLED
        self.max_concurrent = config.LLM_MAX_CONCURRENT
        self._request_queue: Optional[LLMRequestQueue] = None
        
        mode = "QUEUE-BASED PARALLEL" if self.queue_enabled else "SEQUENTIAL BATCH"
        logger.info(f"HierarchicalChunker initialized ({mode} mode, llm={self.llm_service.model})")
        logger.info(f"   chunk_size={self.layer_chunk_size}, summary_size={self.summary_target_size}")
        logger.info(f"   threshold={self.layer_threshold}, max_concurrent={self.max_concurrent}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def split_into_token_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size tokens."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        logger.debug(f"Split text into {len(chunks)} chunks of ~{chunk_size} tokens")
        return chunks
    
    async def extract_clauses_and_risks(self, text: str, chunk_id: str, layer: int, chunk_index: int, context: str = "") -> Dict[str, Any]:
        """
        Extract clauses and analyze risks from a text chunk using LLM.
        Also extracts relationships between clauses.
        
        Args:
            text: Text chunk to analyze
            chunk_id: Identifier for the chunk
            context: Optional context from surrounding chunks
        
        Returns:
            Dictionary containing clauses, risk_level, relationships, and analysis
        """
        system_prompt = get_clause_extraction_system_prompt(layer, chunk_index)

        context_text = f"\n\nContext from document: {context}" if context else ""
        user_prompt = f"Analyze this legal document section (ID: {chunk_id}):{context_text}\n\nText to analyze:\n{text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.llm_service.chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=5000
            )
            
            import json
            import re
            
            with open(f"/tmp/llm_response_{chunk_id}.txt", "w") as f:
                f.write(f"=== RAW RESPONSE FOR {chunk_id} ===\n")
                f.write(response)
                f.write("\n\n=== END RESPONSE ===\n")
            
            response_clean = response.strip()
            
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            elif response_clean.startswith('```'):
                response_clean = response_clean[3:]
            
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            
            response_clean = response_clean.strip()
            
            json_match = re.search(r'\{[\s\S]*\}', response_clean)
            if json_match:
                response_clean = json_match.group(0)
            
            response_clean = response_clean.replace('\n', ' ').replace('\r', ' ')
            response_clean = re.sub(r'\s+', ' ', response_clean)
            
            if not response_clean.endswith('}'):
                logger.warning(f"Incomplete JSON for {chunk_id}, attempting to close")
                open_braces = response_clean.count('{') - response_clean.count('}')
                open_brackets = response_clean.count('[') - response_clean.count(']')
                
                if response_clean.rstrip().endswith(','):
                    response_clean = response_clean.rstrip()[:-1]
                
                if '"risk_summary":' in response_clean and not response_clean.rstrip().endswith('"'):
                    response_clean += '"'
                
                for _ in range(open_brackets):
                    response_clean += ']'
                for _ in range(open_braces):
                    response_clean += '}'
            
            try:
                result = json.loads(response_clean)
            except json.JSONDecodeError as je:
                logger.warning(f"JSON decode error for {chunk_id}: {je}")
                logger.debug(f"Attempted to parse: {response_clean[:500]}")
                
                with open(f"/tmp/llm_response_{chunk_id}_cleaned.txt", "w") as f:
                    f.write(response_clean)
                
                result = {
                    "clauses": [],
                    "clause_relationships": [],
                    "risk_level": "no_risk",
                    "risk_summary": "JSON parsing failed",
                    "key_points": []
                }
            
            if not isinstance(result.get('clauses'), list):
                result['clauses'] = []
            if not isinstance(result.get('clause_relationships'), list):
                result['clause_relationships'] = []
            
            # Validate relationships - filter out invalid references
            valid_clause_ids = {clause.get('clause_id') for clause in result.get('clauses', [])}
            valid_relationships = []
            
            for rel in result.get('clause_relationships', []):
                source_id = rel.get('source_clause_id')
                target_id = rel.get('target_clause_id')
                
                if source_id in valid_clause_ids and target_id in valid_clause_ids:
                    valid_relationships.append(rel)
                else:
                    logger.warning(f"Filtered invalid relationship in {chunk_id}: {source_id} -> {target_id} "
                                 f"(valid clauses: {valid_clause_ids})")
            
            result['clause_relationships'] = valid_relationships
            
            logger.debug(f"Extracted {len(result.get('clauses', []))} clauses and "
                        f"{len(valid_relationships)} valid relationships from chunk {chunk_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting clauses from chunk {chunk_id}: {e}")
            logger.debug(f"Response was: {response[:300] if 'response' in locals() else 'N/A'}")
            return {
                "clauses": [],
                "clause_relationships": [],
                "risk_level": "no_risk",
                "risk_summary": "Analysis failed",
                "key_points": []
            }
    
    async def create_layer1_chunks(self, text: str) -> List[Chunk]:
        """
        Create Layer 1 chunks from the original text.
        OPTIMIZATION: Skip LLM clause extraction at L1 - do it only at L2 to reduce API calls.
        
        Args:
            text: Full document text
        
        Returns:
            List of Layer 1 Chunk objects
        """
        total_tokens = self.count_tokens(text)
        logger.info(f"Creating Layer 1 chunks from {total_tokens} tokens")
        
        text_chunks = self.split_into_token_chunks(text, self.layer_chunk_size)
        num_chunks = len(text_chunks)
        logger.info(f"Split into {num_chunks} chunks of ~{self.layer_chunk_size} tokens each")
        
        layer1_chunks = []
        
        # OPTIMIZATION: Create L1 chunks without LLM calls - just split the text
        # Clause extraction will happen at L2 level to reduce API calls
        for idx, chunk_text in tqdm(enumerate(text_chunks), 
                                     total=num_chunks, 
                                     desc="📄 Creating L1 chunks",
                                     unit="chunk",
                                     colour="blue"):
            chunk_id = f"L1_chunk_{idx}"
            token_count = self.count_tokens(chunk_text)
            
            chunk = Chunk(
                layer=1,
                chunk_id=chunk_id,
                text=chunk_text,
                token_count=token_count,
                clauses=[],  # Will be populated at L2
                risk_level="no_risk",  # Will be determined at L2
                clause_relationships=[],
                metadata={
                    "risk_summary": "",
                    "key_points": [],
                    "position": idx
                }
            )
            
            layer1_chunks.append(chunk)
            logger.debug(f"Created {chunk_id} with {token_count} tokens")
        
        logger.info(f"Created {len(layer1_chunks)} Layer 1 chunks (no LLM calls - optimized)")
        return layer1_chunks
    
    async def _create_short_summaries(self, chunks: List[Chunk]) -> None:
        """
        Create short summary for each chunk (L2+ only, size from config.SUMMARY_TARGET_SIZE).
        These summaries are used as context when creating higher layers.
        
        Args:
            chunks: List of chunks to create short summaries for
        """
        logger.info(f"Creating short summaries for {len(chunks)} chunks")
        
        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            logger.info(f"Processing summary batch {batch_start//self.batch_size + 1}: "
                       f"chunks {batch_start} to {batch_end-1}")
            
            tasks = []
            for chunk in batch_chunks:
                summary_prompt = get_short_summary_prompt(self.summary_target_size)
                user_prompt = f"""Chunk text:
{chunk.text}

Clauses: {chunk.clauses}
Risk: {chunk.risk_level}
Relationships: {chunk.clause_relationships}"""
                
                messages = [
                    {"role": "system", "content": SHORT_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt + "\n\n" + user_prompt}
                ]
                
                logger.info(f"🔵 SHORT_SUMMARY LLM Call for {chunk.chunk_id}")
                logger.info(f"   Prompt length: {len(summary_prompt)} chars")
                logger.info(f"   Target tokens: {self.summary_target_size} (guideline only)")
                logger.debug(f"   Full prompt: {summary_prompt[:500]}...")
                
                tasks.append(self.llm_service.chat_completion(
                    messages=messages,
                    temperature=0.1
                ))
            
            summaries = await asyncio.gather(*tasks, return_exceptions=True)
            
            for chunk, summary_text in zip(batch_chunks, summaries):
                if isinstance(summary_text, Exception):
                    logger.error(f"❌ SHORT_SUMMARY Error for {chunk.chunk_id}: {summary_text}")
                    chunk.short_summary = f"Summary of {chunk.chunk_id} covering {len(chunk.clauses)} clauses."
                else:
                    logger.info(f"✅ SHORT_SUMMARY Response for {chunk.chunk_id}")
                    logger.info(f"   Response length: {len(summary_text)} chars")
                    logger.info(f"   Response tokens: {self.count_tokens(summary_text)}")
                    logger.info(f"   Response preview: {summary_text[:200]}...")
                    if len(summary_text.strip()) == 0:
                        logger.error(f"⚠️  EMPTY RESPONSE for {chunk.chunk_id}!")
                    chunk.short_summary = summary_text
                    logger.debug(f"Created short_summary for {chunk.chunk_id}: "
                               f"{self.count_tokens(summary_text)} tokens")
        
        logger.info(f"Completed short summaries for {len(chunks)} chunks")
    
    def _get_request_queue(self) -> LLMRequestQueue:
        """Get or create the LLM request queue."""
        if self._request_queue is None:
            self._request_queue = LLMRequestQueue(
                llm_service=self.llm_service,
                max_workers=self.max_concurrent
            )
        return self._request_queue
    
    def _parse_l2_json_response(self, response: str, chunk_id: str, fallback_text: str) -> Dict[str, Any]:
        """
        Parse JSON response from L2 LLM call with robust error handling.
        
        Args:
            response: Raw LLM response
            chunk_id: Chunk ID for logging
            fallback_text: Text to use if parsing fails
        
        Returns:
            Parsed dict with summary, clauses, risk_level, etc.
        """
        try:
            response_clean = response.strip()
            
            if response_clean.startswith('```'):
                response_clean = re.sub(r'^```json?\s*', '', response_clean)
                response_clean = re.sub(r'\s*```$', '', response_clean)
            
            json_match = re.search(r'\{[\s\S]*\}', response_clean)
            if json_match:
                response_clean = json_match.group(0)
            
            response_clean = response_clean.replace('\n', ' ').replace('\r', ' ')
            response_clean = re.sub(r'\s+', ' ', response_clean)
            response_clean = re.sub(r',\s*]', ']', response_clean)
            response_clean = re.sub(r',\s*}', '}', response_clean)
            
            open_braces = response_clean.count('{') - response_clean.count('}')
            open_brackets = response_clean.count('[') - response_clean.count(']')
            
            if open_brackets > 0:
                response_clean += ']' * open_brackets
            if open_braces > 0:
                response_clean += '}' * open_braces
            
            parsed = json.loads(response_clean)
            return {
                "summary": parsed.get("summary", fallback_text[:self.layer_chunk_size]),
                "clauses": parsed.get("clauses", []),
                "risk_level": parsed.get("risk_level", "no_risk"),
                "risk_summary": parsed.get("risk_summary", ""),
                "key_points": parsed.get("key_points", []),
                "parse_success": True
            }
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"JSON parse error for {chunk_id}: {e}")
            return {
                "summary": response if response else fallback_text[:self.layer_chunk_size],
                "clauses": [],
                "risk_level": "no_risk",
                "risk_summary": "",
                "key_points": [],
                "parse_success": False
            }
    
    def _build_l2_prompt(self, l1_chunk: Chunk, chunk_index: int) -> List[Dict[str, str]]:
        """Build the prompt messages for L2 processing."""
        combined_prompt = get_l2_summary_prompt(chunk_index, self.summary_target_size)
        combined_prompt += f"\n\nDocument section:\n{l1_chunk.text}"

        return [
            {"role": "system", "content": L2_PROCESSING_SYSTEM_PROMPT},
            {"role": "user", "content": combined_prompt}
        ]
    
    def _create_l2_chunk_from_response(
        self, 
        l1_chunk: Chunk, 
        chunk_index: int, 
        parsed: Dict[str, Any]
    ) -> Chunk:
        """Create an L2 Chunk from parsed LLM response."""
        chunk_id = f"L2_C{chunk_index}"
        summary_text = parsed["summary"]
        clauses = parsed["clauses"]
        risk_level = parsed["risk_level"]
        risk_summary = parsed["risk_summary"]
        key_points = parsed["key_points"]
        
        for clause in clauses:
            clause["source_chunk_id"] = chunk_id
            if "parent_clause_id" not in clause:
                clause["parent_clause_id"] = None
            if "sub_clauses" not in clause:
                clause["sub_clauses"] = []
        
        clause_index = {}
        for clause in clauses:
            clause_id_key = clause.get("clause_id", "")
            if clause_id_key:
                clause_index[clause_id_key] = {
                    "source_chunk_id": chunk_id,
                    "clause_type": clause.get("clause_type", ""),
                    "parent_clause_id": clause.get("parent_clause_id"),
                    "importance": clause.get("importance", "medium"),
                    "layer": 2
                }
        
        main_clauses = [c for c in clauses if c.get("parent_clause_id") is None]
        sub_clauses = [c for c in clauses if c.get("parent_clause_id") is not None]
        
        token_count = self.count_tokens(summary_text)
        
        chunk = Chunk(
            layer=2,
            chunk_id=chunk_id,
            text=summary_text,
            token_count=token_count,
            clauses=clauses,
            risk_level=risk_level,
            clause_relationships=[],
            metadata={
                "risk_summary": risk_summary,
                "key_points": key_points,
                "position": chunk_index,
                "source_l1_chunk": l1_chunk.chunk_id,
                "main_clause_count": len(main_clauses),
                "sub_clause_count": len(sub_clauses)
            },
            parent_id=l1_chunk.chunk_id,
            short_summary=summary_text[:self.summary_target_size],
            clause_index=clause_index
        )
        
        l1_chunk.child_ids.append(chunk_id)
        return chunk

    async def create_layer2_summaries(self, layer1_chunks: List[Chunk]) -> List[Chunk]:
        """
        Create Layer 2 summaries from Layer 1 chunks with 1:1 mapping.
        
        OPTIMIZED: Uses queue-based parallel processing for ~4x speedup.
        - All L1 chunks processed in parallel (controlled by semaphore)
        - Request ID tracking for result retrieval
        - Automatic retry with exponential backoff
        
        Args:
            layer1_chunks: List of Layer 1 chunks
        
        Returns:
            List of Layer 2 Chunk objects (1:1 summaries of L1 chunks)
        """
        num_chunks = len(layer1_chunks)
        logger.info(f"🚀 Creating Layer 2 summaries for {num_chunks} L1 chunks")
        logger.info(f"   Mode: {'QUEUE-BASED PARALLEL' if self.queue_enabled else 'SEQUENTIAL BATCH'}")
        logger.info(f"   Max concurrent: {self.max_concurrent}")
        
        if not self.queue_enabled:
            return await self._create_layer2_summaries_legacy(layer1_chunks)
        
        request_queue = self._get_request_queue()
        requests = []
        chunk_map = {}
        
        for l1_chunk in layer1_chunks:
            chunk_index = l1_chunk.metadata['position']
            request_id = f"L2_C{chunk_index}"
            messages = self._build_l2_prompt(l1_chunk, chunk_index)
            requests.append((request_id, messages))
            chunk_map[request_id] = l1_chunk
        
        logger.info(f"📤 Submitting {len(requests)} L2 requests to queue...")
        # #region agent log
        try:
            import json as _json, time as _time
            with open("/Users/amarchoudhary/Desktop/Legal_CandexAI/.cursor/debug-e24b7b.log", "a", encoding="utf-8") as _df:
                _json.dump({
                    "sessionId": "e24b7b",
                    "hypothesisId": "C",
                    "location": "chunking.py:create_layer2_summaries:start",
                    "message": "L2 batch starting",
                    "data": {"num_requests": len(requests), "max_concurrent": self.max_concurrent},
                    "timestamp": int(_time.time() * 1000),
                }, _df)
                _df.write("\n")
        except Exception:
            pass
        # #endregion
        start_time = time.time()
        
        results = await request_queue.process_batch(
            requests=requests,
            temperature=0.1,
            max_tokens=config.L2_MAX_OUTPUT_TOKENS
        )
        
        elapsed = time.time() - start_time
        logger.info(f"⏱️  L2 parallel processing completed in {elapsed:.1f}s ({num_chunks/elapsed:.1f} chunks/sec)")
        
        layer2_chunks = []
        success_count = 0
        fail_count = 0
        
        sorted_request_ids = sorted(results.keys(), key=lambda x: int(x.split('_C')[1]))
        
        for request_id in tqdm(sorted_request_ids, 
                               desc="📦 Building L2 chunks", 
                               unit="chunk",
                               colour="cyan"):
            response, error = results[request_id]
            l1_chunk = chunk_map[request_id]
            chunk_index = l1_chunk.metadata['position']
            
            if response is None:
                logger.warning(f"❌ {request_id} failed: {error}")
                chunk = Chunk(
                    layer=2,
                    chunk_id=request_id,
                    text=l1_chunk.text,
                    token_count=l1_chunk.token_count,
                    clauses=[],
                    risk_level="no_risk",
                    clause_relationships=[],
                    metadata={"position": chunk_index, "source_l1_chunk": l1_chunk.chunk_id, "retry_failed": True},
                    parent_id=l1_chunk.chunk_id,
                    short_summary=l1_chunk.text[:self.summary_target_size]
                )
                l1_chunk.child_ids.append(request_id)
                layer2_chunks.append(chunk)
                fail_count += 1
            else:
                parsed = self._parse_l2_json_response(response, request_id, l1_chunk.text)
                chunk = self._create_l2_chunk_from_response(l1_chunk, chunk_index, parsed)
                layer2_chunks.append(chunk)
                success_count += 1
                
                if parsed["parse_success"]:
                    logger.debug(f"✅ {request_id}: {len(parsed['clauses'])} clauses, risk={parsed['risk_level']}")
        
        logger.info(f"✅ Created {len(layer2_chunks)} L2 chunks ({success_count} success, {fail_count} failed)")
        return layer2_chunks
    
    async def _create_layer2_summaries_legacy(self, layer1_chunks: List[Chunk]) -> List[Chunk]:
        """
        Legacy sequential batch processing for L2 summaries.
        Used when LLM_QUEUE_ENABLED=False.
        """
        logger.info(f"Using legacy sequential batch processing (batch_size={self.batch_size})")
        
        layer2_chunks = []
        num_chunks = len(layer1_chunks)
        
        for batch_start in range(0, num_chunks, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_chunks)
            batch_chunks = layer1_chunks[batch_start:batch_end]
            
            logger.info(f"Processing L2 batch {batch_start//self.batch_size + 1}: "
                       f"chunks {batch_start} to {batch_end-1}")
            
            for l1_chunk in batch_chunks:
                chunk_index = l1_chunk.metadata['position']
                chunk_id = f"L2_C{chunk_index}"
                messages = self._build_l2_prompt(l1_chunk, chunk_index)
                
                max_retries = 3
                response = None
                
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            await asyncio.sleep(2 ** attempt)
                        
                        response = await self.llm_service.chat_completion(
                            messages=messages,
                            temperature=0.1,
                            max_tokens=config.L2_MAX_OUTPUT_TOKENS
                        )
                        
                        if response and not response.strip().startswith('<!DOCTYPE'):
                            break
                        response = None
                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1} failed for {chunk_id}: {e}")
                        response = None
                
                if response is None:
                    chunk = Chunk(
                        layer=2,
                        chunk_id=chunk_id,
                        text=l1_chunk.text,
                        token_count=l1_chunk.token_count,
                        clauses=[],
                        risk_level="no_risk",
                        clause_relationships=[],
                        metadata={"position": chunk_index, "source_l1_chunk": l1_chunk.chunk_id, "retry_failed": True},
                        parent_id=l1_chunk.chunk_id,
                        short_summary=l1_chunk.text[:self.summary_target_size]
                    )
                else:
                    parsed = self._parse_l2_json_response(response, chunk_id, l1_chunk.text)
                    chunk = self._create_l2_chunk_from_response(l1_chunk, chunk_index, parsed)
                
                layer2_chunks.append(chunk)
        
        logger.info(f"Created {len(layer2_chunks)} Layer 2 summaries (legacy mode)")
        return layer2_chunks
    
    def _group_chunks_by_threshold(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """
        Group chunks by token threshold for creating group summaries.
        
        Args:
            chunks: List of chunks to group
        
        Returns:
            List of chunk groups
        """
        groups = []
        current_group = []
        current_tokens = 0
        
        for chunk in chunks:
            current_group.append(chunk)
            current_tokens += chunk.token_count
            
            if current_tokens >= self.layer_threshold:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    
    async def create_higher_layer_summaries(
        self, 
        lower_layer_chunks: List[Chunk],
        target_layer: int
    ) -> List[Chunk]:
        """
        Create higher layer summaries (L3, L4, etc.) with context-aware processing.
        
        OPTIMIZED: Uses queue-based parallel processing for all groups.
        - All groups processed in parallel (controlled by semaphore)
        - Request ID tracking for result retrieval
        - Automatic retry with exponential backoff
        
        Args:
            lower_layer_chunks: Chunks from the previous layer
            target_layer: Target layer number (3, 4, etc.)
        
        Returns:
            List of higher layer Chunk objects
        """
        logger.info(f"🚀 Creating Layer {target_layer} summaries from {len(lower_layer_chunks)} "
                   f"L{target_layer-1} chunks")
        logger.info(f"   Mode: {'QUEUE-BASED PARALLEL' if self.queue_enabled else 'SEQUENTIAL'}")
        
        # When few chunks remain, merge into one group so hierarchy converges to a root chunk
        if len(lower_layer_chunks) <= 4:
            groups = [lower_layer_chunks]
            logger.info(f"   Final consolidation: merging {len(lower_layer_chunks)} chunks into 1 group")
        else:
            groups = self._group_chunks_by_threshold(lower_layer_chunks)
        num_groups = len(groups)
        logger.info(f"   Grouped into {num_groups} groups")

        if not self.queue_enabled or num_groups <= 1:
            return await self._create_higher_layer_summaries_legacy(lower_layer_chunks, target_layer, groups)
        
        request_queue = self._get_request_queue()
        requests = []
        group_data = {}
        
        for group_idx, group in enumerate(groups):
            request_id = f"L{target_layer}_summary_{group_idx}"
            
            context_summaries = []
            for chunk in group:
                if chunk.short_summary:
                    context_summaries.append(f"Summary of {chunk.chunk_id}:\n{chunk.short_summary}")
            group_context = "\n\n---\n\n".join(context_summaries) if context_summaries else "No context available"
            
            combined_text = "\n\n---\n\n".join([c.text for c in group])
            
            messages = self._build_higher_layer_prompt(group, target_layer, combined_text, group_context)
            requests.append((request_id, messages))
            
            group_data[request_id] = {
                "group": group,
                "group_idx": group_idx,
                "combined_text": combined_text,
                "group_context": group_context
            }
        
        logger.info(f"📤 Submitting {len(requests)} L{target_layer} requests to queue...")
        start_time = time.time()
        
        results = await request_queue.process_batch(
            requests=requests,
            temperature=0.1,
            max_tokens=config.L2_MAX_OUTPUT_TOKENS
        )
        
        elapsed = time.time() - start_time
        logger.info(f"⏱️  L{target_layer} parallel processing completed in {elapsed:.1f}s")
        
        higher_layer_chunks = []
        
        for request_id in sorted(results.keys(), key=lambda x: int(x.split('_')[-1])):
            response, error = results[request_id]
            data = group_data[request_id]
            group = data["group"]
            group_idx = data["group_idx"]
            combined_text = data["combined_text"]
            
            if response is None:
                logger.warning(f"❌ {request_id} failed: {error}, using short summaries as fallback")
                parts = [c.short_summary or c.text[:500] for c in group]
                summary_text = "\n\n".join(parts)
            else:
                summary_text = response.strip() if response.strip() else combined_text
            
            chunk = self._create_higher_layer_chunk(
                group, target_layer, group_idx, summary_text
            )
            higher_layer_chunks.append(chunk)
            
            for child_chunk in group:
                child_chunk.child_ids.append(chunk.chunk_id)
        
        logger.info(f"✅ Created {len(higher_layer_chunks)} Layer {target_layer} chunks")
        return higher_layer_chunks
    
    def _build_higher_layer_prompt(
        self, 
        group: List[Chunk], 
        layer: int, 
        combined_text: str, 
        group_context: str
    ) -> List[Dict[str, str]]:
        """Build prompt messages for higher layer summary."""
        summary_prompt = f"""Summarize the following combined legal document sections. Target length: approximately {self.summary_target_size} tokens (this is a guideline - complete your summary naturally).

This is a Layer {layer} summary combining {len(group)} sections from Layer {layer-1}.

Include:
1. Overall themes and key provisions
2. Critical risks and obligations
3. Important clause relationships
4. Any cross-references or dependencies

Provide the summary directly without any preamble.

Combined text from {len(group)} sections:
{combined_text}

Context from source summaries:
{group_context}"""

        return [
            {"role": "system", "content": "You are a legal document summarizer. Provide clear, high-level overviews."},
            {"role": "user", "content": summary_prompt}
        ]
    
    def _create_higher_layer_chunk(
        self, 
        group: List[Chunk], 
        layer: int, 
        group_idx: int, 
        summary_text: str
    ) -> Chunk:
        """Create a higher layer chunk from group and summary."""
        chunk_id = f"L{layer}_summary_{group_idx}"
        
        all_clauses = []
        combined_clause_index = {}
        risk_levels = []
        
        for c in group:
            for clause in c.clauses:
                clause_copy = clause.copy()
                if "available_in_layers" not in clause_copy:
                    clause_copy["available_in_layers"] = [c.layer]
                else:
                    clause_copy["available_in_layers"] = clause_copy["available_in_layers"] + [c.layer]
                clause_copy["available_in_layers"].append(layer)
                all_clauses.append(clause_copy)
            
            for clause_id_key, clause_info in c.clause_index.items():
                if clause_id_key not in combined_clause_index:
                    combined_clause_index[clause_id_key] = clause_info.copy()
                    combined_clause_index[clause_id_key]["available_in_layers"] = [clause_info.get("layer", c.layer)]
                if layer not in combined_clause_index[clause_id_key].get("available_in_layers", []):
                    combined_clause_index[clause_id_key]["available_in_layers"].append(layer)
            
            if c.risk_level != "no_risk":
                risk_levels.append(c.risk_level)
        
        highest_risk = self._determine_highest_risk(risk_levels)
        token_count = self.count_tokens(summary_text)
        
        main_clauses = [c for c in all_clauses if c.get("parent_clause_id") is None]
        sub_clauses = [c for c in all_clauses if c.get("parent_clause_id") is not None]
        
        return Chunk(
            layer=layer,
            chunk_id=chunk_id,
            text=summary_text,
            token_count=token_count,
            clauses=all_clauses,
            risk_level=highest_risk,
            metadata={
                "summarized_from": [c.chunk_id for c in group],
                "num_source_chunks": len(group),
                "group_index": group_idx,
                "total_clauses": len(all_clauses),
                "main_clause_count": len(main_clauses),
                "sub_clause_count": len(sub_clauses)
            },
            parent_id=None,
            short_summary=summary_text[:self.summary_target_size],
            clause_index=combined_clause_index
        )
    
    async def _create_higher_layer_summaries_legacy(
        self, 
        lower_layer_chunks: List[Chunk],
        target_layer: int,
        groups: List[List[Chunk]]
    ) -> List[Chunk]:
        """Legacy sequential processing for higher layer summaries."""
        higher_layer_chunks = []
        
        for group_idx, group in enumerate(groups):
            context_summaries = []
            for chunk in group:
                if chunk.short_summary:
                    context_summaries.append(f"Summary of {chunk.chunk_id}:\n{chunk.short_summary}")
            
            group_context = "\n\n---\n\n".join(context_summaries) if context_summaries else "No context available"
            
            summary_chunk = await self._summarize_group_with_context(
                group, 
                target_layer, 
                group_idx,
                group_context
            )
            higher_layer_chunks.append(summary_chunk)
            
            for child_chunk in group:
                child_chunk.child_ids.append(summary_chunk.chunk_id)
        
        for chunk in higher_layer_chunks:
            if not chunk.short_summary:
                chunk.short_summary = chunk.text[:self.summary_target_size]
        
        logger.info(f"Created {len(higher_layer_chunks)} Layer {target_layer} summaries (legacy mode)")
        return higher_layer_chunks
    
    async def _summarize_group_with_context(
        self, 
        chunks: List[Chunk], 
        layer: int, 
        group_idx: int,
        group_context: str
    ) -> Chunk:
        """Summarize a group of chunks with context from the group summary."""
        chunk_id = f"L{layer}_summary_{group_idx}"
        
        combined_text = "\n\n---\n\n".join([c.text for c in chunks])
        
        # Aggregate all clauses with source tracking
        all_clauses = []
        combined_clause_index = {}
        risk_levels = []
        
        for c in chunks:
            # Add each clause with its original source preserved
            for clause in c.clauses:
                # Clause already has source_chunk_id from L2 extraction
                # Add available_in_layers to track which layers contain this clause
                clause_copy = clause.copy()
                if "available_in_layers" not in clause_copy:
                    clause_copy["available_in_layers"] = [c.layer]
                else:
                    clause_copy["available_in_layers"] = clause_copy["available_in_layers"] + [c.layer]
                clause_copy["available_in_layers"].append(layer)  # Add current layer
                all_clauses.append(clause_copy)
            
            # Merge clause_index from child chunks
            for clause_id, clause_info in c.clause_index.items():
                if clause_id not in combined_clause_index:
                    combined_clause_index[clause_id] = clause_info.copy()
                    combined_clause_index[clause_id]["available_in_layers"] = [clause_info.get("layer", c.layer)]
                # Add current layer to available_in_layers
                if layer not in combined_clause_index[clause_id].get("available_in_layers", []):
                    combined_clause_index[clause_id]["available_in_layers"].append(layer)
            
            if c.risk_level != "no_risk":
                risk_levels.append(c.risk_level)
        
        highest_risk = self._determine_highest_risk(risk_levels)
        
        # Log clause aggregation
        main_clauses = [c for c in all_clauses if c.get("parent_clause_id") is None]
        sub_clauses = [c for c in all_clauses if c.get("parent_clause_id") is not None]
        logger.info(f"   Aggregating {len(all_clauses)} clauses ({len(main_clauses)} main, {len(sub_clauses)} sub) from {len(chunks)} chunks")
        
        summary_prompt = f"""Summarize the following combined legal document sections. Target length: approximately {self.summary_target_size} tokens (this is a guideline - complete your summary naturally).

This is a Layer {layer} summary combining {len(chunks)} sections from Layer {layer-1}.

Include:
1. Overall themes and key provisions
2. Critical risks and obligations
3. Important clause relationships
4. Any cross-references or dependencies

Provide the summary directly without any preamble.

Combined text from {len(chunks)} sections:
{combined_text}

Context from source summaries:
{group_context}"""

        messages = [
            {"role": "system", "content": "You are a legal document summarizer. Provide clear, high-level overviews."},
            {"role": "user", "content": summary_prompt}
        ]
        
        logger.info(f"🔵 L{layer}_SUMMARY LLM Call for {chunk_id}")
        logger.info(f"   Combining {len(chunks)} chunks from L{layer-1}")
        logger.info(f"   Combined text length: {len(combined_text)} chars")
        logger.info(f"   Context length: {len(group_context)} chars")
        logger.info(f"   Prompt length: {len(summary_prompt)} chars")
        logger.info(f"   Target tokens: ~{self.summary_target_size} (guideline only)")
        logger.debug(f"   Combined text preview: {combined_text[:300]}...")
        
        try:
            # Retry logic: up to 3 attempts with exponential backoff
            max_retries = 3
            summary_text = None
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        wait_time = 2 ** attempt  # 2, 4, 8 seconds
                        logger.info(f"   🔄 Retry {attempt}/{max_retries-1} for {chunk_id} after {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    
                    summary_text = await self.llm_service.chat_completion(
                        messages=messages,
                        temperature=0.1
                    )
                    
                    # Check if response is valid (not empty, not HTML error page)
                    if summary_text and not summary_text.strip().startswith('<!DOCTYPE') and not summary_text.strip().startswith('<html'):
                        break  # Success
                    else:
                        last_error = "Empty or HTML error response"
                        logger.warning(f"   ⚠️ Invalid response for {chunk_id}: {last_error}")
                        summary_text = None
                        
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"   ⚠️ Attempt {attempt+1}/{max_retries} failed for {chunk_id}: {e}")
                    summary_text = None
            
            # Use fallback if all retries failed
            if summary_text is None:
                logger.error(f"❌ L{layer}_SUMMARY Failed after {max_retries} retries for {chunk_id}: {last_error}")
                summary_text = combined_text  # Fallback to combined text
            
            logger.info(f"✅ L{layer}_SUMMARY Response for {chunk_id}")
            logger.info(f"   Response length: {len(summary_text)} chars")
            logger.info(f"   Response tokens: {self.count_tokens(summary_text)}")
            logger.info(f"   Response preview: {summary_text[:200]}...")
            
            # FALLBACK: If LLM returns empty, use combined source text instead
            if len(summary_text.strip()) == 0:
                logger.warning(f"⚠️  EMPTY RESPONSE for {chunk_id}! Using combined source text as fallback.")
                logger.warning(f"   Source chunks: {[c.chunk_id for c in chunks]}")
                logger.warning(f"   Falling back to combined text ({len(combined_text)} chars)")
                summary_text = combined_text
            
            token_count = self.count_tokens(summary_text)
            
            # Count main vs sub clauses for metadata
            final_main_clauses = [c for c in all_clauses if c.get("parent_clause_id") is None]
            final_sub_clauses = [c for c in all_clauses if c.get("parent_clause_id") is not None]
            clause_type_counts: Dict[str, int] = {}
            for _cid, info in combined_clause_index.items():
                ct = info.get("clause_type", "Unknown")
                clause_type_counts[ct] = clause_type_counts.get(ct, 0) + 1

            chunk = Chunk(
                layer=layer,
                chunk_id=chunk_id,
                text=summary_text,
                token_count=token_count,
                clauses=all_clauses,
                risk_level=highest_risk,
                metadata={
                    "summarized_from": [c.chunk_id for c in chunks],
                    "num_source_chunks": len(chunks),
                    "group_index": group_idx,
                    "total_clauses": len(all_clauses),
                    "main_clause_count": len(final_main_clauses),
                    "sub_clause_count": len(final_sub_clauses),
                    "clause_type_counts": clause_type_counts,
                },
                parent_id=None,
                short_summary=summary_text[:self.summary_target_size],  # OPTIMIZATION: Set short_summary during creation
                clause_index=combined_clause_index
            )
            
            logger.info(f"   Clause index built: {len(combined_clause_index)} entries")
            
            logger.debug(f"Created {chunk_id} from {len(chunks)} chunks, "
                        f"tokens={token_count}, risk={highest_risk}")
            return chunk
            
        except Exception as e:
            logger.error(f"Error creating L{layer} summary: {e}")
            raise
    
    def _determine_highest_risk(self, risk_levels: List[str]) -> str:
        """Determine the highest risk level from a list of risk levels."""
        if not risk_levels:
            return "no_risk"
        
        risk_priority = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "no_risk": 1
        }
        
        highest = max(risk_levels, key=lambda x: risk_priority.get(x, 0))
        return highest
    
    async def process_document(self, text: str) -> Dict[str, Any]:
        """
        Process a document through the complete hierarchical chunking pipeline.
        
        Args:
            text: Full document text
        
        Returns:
            Dictionary containing all layers of chunks and metadata
        """
        logger.info("Starting hierarchical chunking process")
        
        total_tokens = self.count_tokens(text)
        logger.info(f"Document has {total_tokens} tokens")
        
        layer1_chunks = await self.create_layer1_chunks(text)
        
        layer2_chunks = await self.create_layer2_summaries(layer1_chunks)
        
        all_layers = {
            "layer_1": layer1_chunks,
            "layer_2": layer2_chunks
        }
        
        current_layer_chunks = layer2_chunks
        current_layer_num = 2
        
        # Continue creating layers until we have a single chunk (root of the tree)
        max_layers = config.MAX_HIERARCHY_LAYERS
        while len(current_layer_chunks) > 1 and current_layer_num < max_layers:
            current_layer_num += 1

            total_tokens_in_layer = sum(c.token_count for c in current_layer_chunks)
            logger.info(
                f"Layer {current_layer_num - 1} has {len(current_layer_chunks)} chunks "
                f"with {total_tokens_in_layer} total tokens"
            )

            next_layer_chunks = await self.create_higher_layer_summaries(
                current_layer_chunks,
                current_layer_num,
            )

            all_layers[f"layer_{current_layer_num}"] = next_layer_chunks
            current_layer_chunks = next_layer_chunks

            logger.info(f"Created Layer {current_layer_num} with {len(next_layer_chunks)} chunks")

            if len(current_layer_chunks) == 1:
                logger.info(
                    f"Reached final layer {current_layer_num} with single root chunk"
                )
                break

        if len(current_layer_chunks) > 1:
            logger.warning(
                f"Stopped at {len(current_layer_chunks)} chunks after layer {current_layer_num} "
                f"(max_layers={max_layers})"
            )
        
        tree_structure = self._build_tree_structure(all_layers)
        
        result = {
            "total_tokens": total_tokens,
            "num_layers": len(all_layers),
            "layers": all_layers,
            "tree_structure": tree_structure,
            "metadata": {
                "layer_chunk_size": self.layer_chunk_size,
                "summary_target_size": self.summary_target_size,
                "layer_threshold": self.layer_threshold,
                "batch_size": self.batch_size,
                "total_chunks": sum(len(chunks) for chunks in all_layers.values())
            }
        }
        
        logger.info(f"Hierarchical chunking complete: {result['num_layers']} layers, "
                   f"{result['metadata']['total_chunks']} total chunks")
        
        return result
    
    def _build_tree_structure(self, all_layers: Dict[str, List[Chunk]]) -> Dict[str, Any]:
        """
        Build a tree structure representation of the chunk hierarchy.
        Includes a global clause_index from the root chunk for retrieval.
        
        Args:
            all_layers: Dictionary of all layers with their chunks
        
        Returns:
            Tree structure with parent-child relationships and global clause index
        """
        tree = {
            "nodes": {},
            "root_nodes": [],
            "relationships": [],
            "global_clause_index": {},  # Combined clause index from root chunk
            "clause_statistics": {}  # Summary statistics
        }
        
        for layer_name, chunks in all_layers.items():
            for chunk in chunks:
                tree["nodes"][chunk.chunk_id] = chunk.to_dict()
                
                if chunk.parent_id:
                    tree["relationships"].append({
                        "parent": chunk.parent_id,
                        "child": chunk.chunk_id,
                        "relationship_type": "summarizes"
                    })
                else:
                    tree["root_nodes"].append(chunk.chunk_id)
                
                for clause_rel in chunk.clause_relationships:
                    tree["relationships"].append({
                        "source": clause_rel.get("source_clause_id"),
                        "target": clause_rel.get("target_clause_id"),
                        "relationship_type": clause_rel.get("relationship_type"),
                        "description": clause_rel.get("description"),
                        "chunk_id": chunk.chunk_id
                    })
        
        # Extract global clause index from root chunk(s)
        # The root chunk contains the aggregated clause_index from all layers
        for root_id in tree["root_nodes"]:
            root_chunk_data = tree["nodes"].get(root_id, {})
            root_clause_index = root_chunk_data.get("clause_index", {})
            tree["global_clause_index"].update(root_clause_index)
        
        # Calculate clause statistics
        all_clauses = []
        for node_id, node_data in tree["nodes"].items():
            all_clauses.extend(node_data.get("clauses", []))
        
        main_clauses = [c for c in all_clauses if c.get("parent_clause_id") is None]
        sub_clauses = [c for c in all_clauses if c.get("parent_clause_id") is not None]
        
        # Get unique clauses from root (final aggregated list)
        root_clauses = []
        for root_id in tree["root_nodes"]:
            root_clauses.extend(tree["nodes"].get(root_id, {}).get("clauses", []))
        
        unique_main = [c for c in root_clauses if c.get("parent_clause_id") is None]
        unique_sub = [c for c in root_clauses if c.get("parent_clause_id") is not None]
        
        clause_type_counts: Dict[str, int] = {}
        for _cid, info in tree["global_clause_index"].items():
            ct = info.get("clause_type", "Unknown")
            clause_type_counts[ct] = clause_type_counts.get(ct, 0) + 1
        tree["clause_type_counts"] = clause_type_counts

        tree["clause_statistics"] = {
            "total_clauses_in_root": len(root_clauses),
            "main_clauses": len(unique_main),
            "sub_clauses": len(unique_sub),
            "clause_index_entries": len(tree["global_clause_index"]),
        }
        
        logger.info(f"Built tree structure with {len(tree['nodes'])} nodes and "
                   f"{len(tree['relationships'])} relationships")
        logger.info(f"Global clause index: {len(tree['global_clause_index'])} entries "
                   f"({tree['clause_statistics']['main_clauses']} main, "
                   f"{tree['clause_statistics']['sub_clauses']} sub)")
        
        return tree
