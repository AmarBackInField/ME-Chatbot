import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, APITimeoutError

_DEBUG_LOG_PATH = "/Users/amarchoudhary/Desktop/Legal_CandexAI/.cursor/debug-e24b7b.log"
_DEBUG_SESSION = "e24b7b"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    # #region agent log
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
        entry = {
            "sessionId": _DEBUG_SESSION,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass
    # #endregion


sys.path.append(os.path.dirname(__file__))
from config import config
from utils.logger import get_logger

logger = get_logger("LLMService")


def _uses_max_completion_tokens(model: str) -> bool:
    """GPT-5 and reasoning models require max_completion_tokens, not max_tokens."""
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _supports_custom_temperature(model: str) -> bool:
    """GPT-5 / reasoning models only accept the default temperature (omit param)."""
    return not _uses_max_completion_tokens(model)


def _build_completion_params(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_output_tokens: int,
) -> dict:
    params: dict = {
        "model": model,
        "messages": messages,
    }
    if _supports_custom_temperature(model):
        params["temperature"] = temperature
    if _uses_max_completion_tokens(model):
        params["max_completion_tokens"] = max_output_tokens
    else:
        params["max_tokens"] = max_output_tokens
    return params


class LLMService:
    """
    LLM Service for interacting with OpenAI API.
    Provides chat completions for legal document analysis.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self._client: Optional[AsyncOpenAI] = None
        logger.info(f"LLMService initialized (model={self.model})")

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=float(config.LLM_TIMEOUT),
            )
        return self._client

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Send a chat completion request to OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (defaults to config)
            max_tokens: Maximum tokens in response (defaults to config)
            stream: Unused; kept for backward compatibility
            model: Optional model override for this request

        Returns:
            Generated text response
        """
        del stream, kwargs

        effective_model = model or self.model
        client = self._get_client()
        temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS

        logger.debug(f"Sending chat completion with {len(messages)} messages (model={effective_model})")

        params = _build_completion_params(effective_model, messages, temp, tokens)
        _agent_debug_log(
            "B",
            "llm_service.py:chat_completion:pre",
            "OpenAI request params",
            {
                "model": effective_model,
                "max_output": tokens,
                "param_keys": list(params.keys()),
                "message_count": len(messages),
                "runId": "post-fix",
            },
        )
        try:
            t0 = time.time()
            response = await client.chat.completions.create(**params)
            elapsed_ms = int((time.time() - t0) * 1000)
            choice = response.choices[0]
            msg = choice.message
            content = msg.content or ""
            usage = getattr(response, "usage", None)
            _agent_debug_log(
                "A",
                "llm_service.py:chat_completion:post",
                "OpenAI response shape",
                {
                    "elapsed_ms": elapsed_ms,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "content_len": len(content),
                    "content_preview": content[:120] if content else "",
                    "refusal": getattr(msg, "refusal", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "reasoning_tokens": (
                        getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
                        if usage
                        else None
                    ),
                    "model": effective_model,
                    "runId": "post-fix",
                },
            )
            logger.debug(f"Received response with {len(content)} characters (model={effective_model})")
            return content
        except APITimeoutError as e:
            _agent_debug_log(
                "B",
                "llm_service.py:chat_completion:timeout",
                "OpenAI timeout",
                {"error": str(e)[:300], "timeout_s": config.LLM_TIMEOUT, "model": effective_model, "runId": "post-fix"},
            )
            logger.error(f"OpenAI chat completion timed out after {config.LLM_TIMEOUT}s: {e}")
            raise TimeoutError(f"OpenAI request timed out after {config.LLM_TIMEOUT}s") from e
        except Exception as e:
            _agent_debug_log(
                "E",
                "llm_service.py:chat_completion:exception",
                "OpenAI exception",
                {"error_type": type(e).__name__, "error": str(e)[:400], "model": effective_model, "runId": "post-fix"},
            )
            logger.error(f"Error in chat completion: {e}")
            raise

    def chat_completion_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Synchronous version of chat_completion."""
        import asyncio

        return asyncio.run(
            self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                **kwargs,
            )
        )

    async def analyze_legal_document(self, document_text: str) -> str:
        """Analyze a legal document for risks and issues."""
        system_prompt = """You are an expert legal document analyst specializing in risk assessment. 
Your task is to analyze legal documents and identify potential risks, problematic clauses, 
and areas of concern. Provide a comprehensive risk analysis with the following structure:

1. **Overall Risk Level**: (Critical/High/Medium/Low)
2. **Executive Summary**: Brief overview of the document and its purpose
3. **Risks Identified**:
   - List each risk with explanation
4. **Problematic Clauses**: Specific clauses that may be unfavorable
5. **Legal Compliance Issues**: Any potential compliance concerns
6. **Recommendations**:
   - Actionable suggestions to mitigate risks

Be thorough, specific, and provide actionable insights."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Please analyze the following legal document for risks:\n\n{document_text}",
            },
        ]

        logger.info("Analyzing legal document for risks")
        return await self.chat_completion(messages)

    async def compare_documents(
        self,
        user_document: str,
        template_document: str,
    ) -> str:
        """Compare a user document with a standard template."""
        system_prompt = """You are an expert legal document analyst specializing in contract comparison.
Your task is to compare a user's legal document against a standard template and identify:
- Key differences between the documents
- Missing clauses in the user's document
- Additional clauses in the user's document not in the template
- Compliance assessment
- Risk implications of the differences

Provide a structured analysis with:
1. **Compliance Score**: (percentage or rating)
2. **Executive Summary**: Brief overview of the comparison
3. **Key Differences**: List significant differences
4. **Missing Clauses**: Clauses in template but not in user document
5. **Additional Clauses**: Clauses in user document but not in template
6. **Risk Assessment**: Implications of the differences
7. **Recommendations**: Specific actions to align with the template"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Compare these documents:\n\nUSER DOCUMENT:\n{user_document}\n\nTEMPLATE:\n{template_document}",
            },
        ]

        logger.info("Comparing documents")
        return await self.chat_completion(messages)


llm_service = LLMService()


async def chat_completion(messages: List[Dict[str, str]], **kwargs) -> str:
    """Convenience function for chat completion."""
    return await llm_service.chat_completion(messages, **kwargs)


async def analyze_legal_document(document_text: str) -> str:
    """Convenience function for legal document analysis."""
    return await llm_service.analyze_legal_document(document_text)


async def compare_documents(user_doc: str, template_doc: str) -> str:
    """Convenience function for document comparison."""
    return await llm_service.compare_documents(user_doc, template_doc)
