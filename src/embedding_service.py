"""
OpenAI embeddings client for vector search.
"""

import asyncio
from typing import List, Optional

from openai import AsyncOpenAI

import sys
import os

sys.path.append(os.path.dirname(__file__))
from config import config
from utils.logger import get_logger

logger = get_logger("EmbeddingService")


class OpenAIEmbeddings:
    """OpenAI embeddings client using text-embedding-3-large."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = config.EMBED_MODEL
        self.dimensions = config.EMBED_DIMENSIONS
        self._async_client: Optional[AsyncOpenAI] = None

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=config.EMBED_TIMEOUT,
            )
        return self._async_client

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        client = self._get_async_client()
        response = await client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single text query (sync)."""
        return asyncio.run(self.async_embed_query(text))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (sync, batched)."""
        if not texts:
            return []
        return asyncio.run(self.async_embed_documents(texts))

    async def async_embed_query(self, text: str, max_retries: int = 3) -> List[float]:
        """Async generate embedding for a single text query with retry logic."""
        last_error = None
        for retry in range(max_retries):
            try:
                results = await self._embed_batch([text])
                return results[0]
            except Exception as e:
                last_error = e
                if retry < max_retries - 1:
                    await asyncio.sleep((retry + 1) * 1.5)
                    continue
        raise last_error

    async def async_embed_documents(
        self, texts: List[str], batch_size: int = 50, max_retries: int = 3
    ) -> List[List[float]]:
        """Async generate embeddings for multiple texts with batching and retry logic."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(texts), batch_size):
            batch = texts[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            for retry in range(max_retries):
                try:
                    batch_embeddings = await self._embed_batch(batch)
                    all_embeddings.extend(batch_embeddings)
                    logger.info(
                        f"Embedded batch {batch_num}/{total_batches} ({len(batch)} texts)"
                    )
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = (retry + 1) * 2
                        logger.warning(
                            f"Batch {batch_num} failed (attempt {retry + 1}/{max_retries}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Batch {batch_num} failed after {max_retries} retries: {e}")
                        raise

        return all_embeddings
