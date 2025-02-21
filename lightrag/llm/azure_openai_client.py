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


class AzureOpenaiClient:
    __llm_model = None
    __llm_endpoint = None
    __llm_api_version = None
    __embedding_model = None
    __embedding_endpoint = None
    __embedding_api_version = None
    __embedding_dimension = None

    def __init__(self, **kwargs):
        pass  #

    @classmethod
    def set_parameters(
        cls,
        llm_model,
        llm_endpoint,
        llm_api_version,
        embedding_model,
        embedding_endpoint,
        embedding_api_version,
        embedding_dimension,
    ):
        cls.__llm_model = llm_model
        cls.__llm_endpoint = llm_endpoint
        cls.__llm_api_version = llm_api_version
        cls.__embedding_api_version = embedding_api_version
        cls.__embedding_model = embedding_model
        cls.__embedding_endpoint = embedding_endpoint
        cls.__embedding_dimension = embedding_dimension

    @classmethod
    def get_instance(cls, **kwargs):
        return cls()

    @classmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APIConnectionError,
            )
        ),
    )
    async def azure_openai_complete_if_cache(
        cls,
        prompt,
        azure_ad_token,
        system_prompt=None,
        history_messages=[],
        **kwargs,
    ):
        openai_async_client = openai.AsyncAzureOpenAI(
            azure_endpoint=cls.__llm_endpoint,
            azure_ad_token=azure_ad_token,
            api_version=cls.__llm_api_version,
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
                model=cls.__llm_model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=cls.__llm_model, messages=messages, **kwargs
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

    @classmethod
    @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8191)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)
        ),
    )
    async def azure_openai_embed(
        cls,
        texts: list[str],
        azure_ad_token: str,
    ) -> np.ndarray:
        openai_async_client = openai.AsyncAzureOpenAI(
            azure_endpoint=cls.__embedding_endpoint,
            azure_ad_token=azure_ad_token,
            api_version=cls.__embedding_api_version,
        )
        response = await openai_async_client.embeddings.create(
            model=cls.__embedding_model,
            input=texts,
            encoding_format="float",
            dimensions=cls.__embedding_dimension,
        )
        return np.array([dp.embedding for dp in response.data])
