"""
Azure OpenAI LLM Interface Module
==========================

This module provides interfaces for interacting with aure openai's language models,
including text generation and embedding capabilities.

Author: Lightrag team
Created: 2024-01-24
License: MIT License

Copyright (c) 2024 Lightrag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

Version: 1.0.0

Change Log:
- 1.0.0 (2024-01-24): Initial release
    * Added async chat completion support
    * Added embedding generation
    * Added stream response capability

Dependencies:
    - openai
    - numpy
    - pipmaster
    - Python >= 3.10

Usage:
    from llm_interfaces.azure_openai import azure_openai_model_complete, azure_openai_embed
"""

__version__ = "1.0.0"
__author__ = "lightrag Team"
__status__ = "Production"

import openai as openai
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
)



def check_model_version(
    model_name: str, endpoint: str, azure_ad_token: str, api_version: str
) -> dict:
    openai.api_base = endpoint
    openai.api_key = azure_ad_token
    openai.api_version = api_version

    try:
        response = openai.Model.retrieve(model_name)
        return {
            "model_name": response.get("id"),
            "created": response.get("created"),
            "version": response.get("version"),
        }
    except Exception as e:
        return {"error": str(e)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIConnectionError, openai.APIConnectionError)
    ),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    endpoint,
    azure_ad_token,
    api_version,
    system_prompt=None,
    history_messages=[],
    **kwargs,
):
    openai_async_client = openai.AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token=azure_ad_token,
        api_version=api_version,
    )
    kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    if "response_format" in kwargs:
        # This requires GPT-4o model with version 2024-08-06 and later
        response = await openai_async_client.beta.chat.completions.parse(
            model=model, messages=messages, **kwargs
        )
    else:
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    if hasattr(response, "__aiter__"):

        async def inner():
            async for chunk in response:
                if len(chunk.choices) == 0:
                    continue
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content

        return inner()
    else:
        content = response.choices[0].message.content
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content


async def azure_openai_complete(
    model,
    prompt,
    endpoint,
    azure_ad_token,
    api_version,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> str:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await azure_openai_complete_if_cache(
        model=model,
        prompt=prompt,
        endpoint=endpoint,
        azure_ad_token=azure_ad_token,
        api_version=api_version,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    if keyword_extraction:  # TODO: use JSON API
        return locate_json_string_body_from_string(result)
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8191)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
    ),
)
async def azure_openai_embed(
    texts: list[str],
    model: str,
    endpoint: str,
    azure_ad_token: str,
    api_version: str,
) -> np.ndarray:
    openai_async_client = openai.AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token=azure_ad_token,
        api_version=api_version,
    )
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
