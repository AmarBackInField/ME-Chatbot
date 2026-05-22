"""
API Utilities
Common utilities for API calls including retry logic with exponential backoff.
"""

import asyncio
import httpx
from typing import Dict, Any, Optional
from functools import wraps

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.logger import get_logger

logger = get_logger("APIUtils")


async def async_api_call_with_retry(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 30.0,
    timeout: float = 180.0
) -> httpx.Response:
    """
    Make an async API call with exponential backoff retry logic.
    
    Args:
        url: API endpoint URL
        payload: Request payload (JSON)
        headers: Request headers
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        timeout: Request timeout (seconds)
    
    Returns:
        httpx.Response object
    
    Raises:
        Exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                
                # Check for server errors that should be retried
                if response.status_code == 504:  # Gateway Timeout
                    raise httpx.TimeoutException(f"Gateway timeout (504)")
                elif response.status_code == 502:  # Bad Gateway
                    raise httpx.TimeoutException(f"Bad gateway (502)")
                elif response.status_code == 503:  # Service Unavailable
                    raise httpx.TimeoutException(f"Service unavailable (503)")
                elif response.status_code >= 500:  # Other server errors
                    raise httpx.HTTPStatusError(
                        f"Server error: {response.status_code}",
                        request=response.request,
                        response=response
                    )
                
                return response
                
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)[:100]}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)  # Exponential backoff
            else:
                logger.error(f"API call failed after {max_retries + 1} attempts: {str(e)[:200]}")
                raise Exception(f"API call failed after {max_retries + 1} attempts: {str(e)}")
        
        except Exception as e:
            # Non-retryable errors
            logger.error(f"Non-retryable API error: {str(e)[:200]}")
            raise
    
    raise Exception(f"API call failed: {last_exception}")


def sync_api_call_with_retry(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 30.0,
    timeout: float = 60.0
) -> Any:
    """
    Make a sync API call with exponential backoff retry logic.
    """
    import requests
    import time
    
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            
            if response.status_code in [502, 503, 504]:
                raise requests.exceptions.Timeout(f"Server error: {response.status_code}")
            elif response.status_code >= 500:
                raise requests.exceptions.HTTPError(f"Server error: {response.status_code}")
            
            return response
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exception = e
            
            if attempt < max_retries:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)[:100]}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                logger.error(f"API call failed after {max_retries + 1} attempts")
                raise Exception(f"API call failed after {max_retries + 1} attempts: {str(e)}")
        
        except Exception as e:
            logger.error(f"Non-retryable API error: {str(e)[:200]}")
            raise
    
    raise Exception(f"API call failed: {last_exception}")
