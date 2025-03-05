"""
This module contains all query-related routes for the LightRAG API.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional
from fastapi.responses import JSONResponse
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Header,
)

from ... import LightRAG
from ...base import QueryParam
from ..utils_api import (
    get_api_key_dependency,
    initialize_rag_with_header,
    wait_for_storage_initialization,
    get_lightrag_token_credential,
    extract_token_value,
)
from pydantic import BaseModel, Field, field_validator
from ascii_colors import trace_exception

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="The query text",
    )

    mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(
        default="hybrid",
        description="Query mode",
    )

    only_need_context: Optional[bool] = Field(
        default=None,
        description="If True, only returns the retrieved context without generating a response.",
    )

    only_need_prompt: Optional[bool] = Field(
        default=None,
        description="If True, only returns the generated prompt without producing a response.",
    )

    response_type: Optional[str] = Field(
        min_length=1,
        default=None,
        description="Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'.",
    )

    top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode.",
    )

    max_token_for_text_unit: Optional[int] = Field(
        gt=1,
        default=None,
        description="Maximum number of tokens allowed for each retrieved text chunk.",
    )

    max_token_for_global_context: Optional[int] = Field(
        gt=1,
        default=None,
        description="Maximum number of tokens allocated for relationship descriptions in global retrieval.",
    )

    max_token_for_local_context: Optional[int] = Field(
        gt=1,
        default=None,
        description="Maximum number of tokens allocated for entity descriptions in local retrieval.",
    )

    hl_keywords: Optional[List[str]] = Field(
        default=None,
        description="List of high-level keywords to prioritize in retrieval.",
    )

    ll_keywords: Optional[List[str]] = Field(
        default=None,
        description="List of low-level keywords to refine retrieval focus.",
    )

    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Stores past conversation history to maintain context. Format: [{'role': 'user/assistant', 'content': 'message'}].",
    )

    history_turns: Optional[int] = Field(
        ge=0,
        default=None,
        description="Number of complete conversation turns (user-assistant pairs) to consider in the response context.",
    )

    @field_validator("query", mode="after")
    @classmethod
    def query_strip_after(cls, query: str) -> str:
        return query.strip()

    @field_validator("hl_keywords", mode="after")
    @classmethod
    def hl_keywords_strip_after(cls, hl_keywords: List[str] | None) -> List[str] | None:
        if hl_keywords is None:
            return None
        return [keyword.strip() for keyword in hl_keywords]

    @field_validator("ll_keywords", mode="after")
    @classmethod
    def ll_keywords_strip_after(cls, ll_keywords: List[str] | None) -> List[str] | None:
        if ll_keywords is None:
            return None
        return [keyword.strip() for keyword in ll_keywords]

    @field_validator("conversation_history", mode="after")
    @classmethod
    def conversation_history_role_check(
        cls, conversation_history: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        if conversation_history is None:
            return None
        for msg in conversation_history:
            if "role" not in msg or msg["role"] not in {"user", "assistant"}:
                raise ValueError(
                    "Each message must have a 'role' key with value 'user' or 'assistant'."
                )
        return conversation_history

    def to_query_params(self, is_stream: bool) -> "QueryParam":
        """Converts a QueryRequest instance into a QueryParam instance."""
        # Use Pydantic's `.model_dump(exclude_none=True)` to remove None values automatically
        request_data = self.model_dump(exclude_none=True, exclude={"query"})

        # Ensure `mode` and `stream` are set explicitly
        param = QueryParam(**request_data)
        param.stream = is_stream
        return param


class QueryResponse(BaseModel):
    response: str = Field(
        description="The generated response",
    )


def create_query_routes(
    rag_instance_manager, api_key: Optional[str] = None, top_k: int = 60
):
    optional_api_key = get_api_key_dependency(api_key)

    @router.post(
        "/query", response_model=QueryResponse, dependencies=[Depends(optional_api_key)]
    )
    async def query_text(
        request: QueryRequest,
        storage_account_url: str = Header(alias="Storage_Account_Url"),
        storage_container_name: str = Header(alias="Storage_Container_Name"),
        storage_token_expiry: str = Header(
            default=None, alias="Storage_Access_Token_Expiry"
        ),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        storage_access_token: str = Header(alias="Storage_Access_Token"),
        X_Affinity_Token: str = Header(None, alias="X-Affinity-Token"),
    ):
        """
        Handle a POST request at the /query endpoint to process user queries using RAG capabilities.
        """
        try:
            ai_access_token = extract_token_value(ai_access_token, "Azure-AI-Access-Token")
            storage_access_token = extract_token_value(
                storage_access_token, "Storage_Access_Token"
            )
            lightrag_token = get_lightrag_token_credential(
                storage_access_token, storage_token_expiry
            )
            param = request.to_query_params(False)
            rag: LightRAG = initialize_rag_with_header(
                rag_instance_manager,
                storage_account_url,
                storage_container_name,
                X_Affinity_Token,
                storage_access_token,
                storage_token_expiry,
            )
            await wait_for_storage_initialization(
                rag,
                get_lightrag_token_credential(
                    storage_access_token, storage_token_expiry
                ),
            )
            response = await rag.aquery(
                request.query,
                ai_access_token,
                storage_account_url,
                storage_container_name,
                lightrag_token,
                param=param
            )

            # If response is a string (e.g. cache hit), return directly
            if isinstance(response, str):
                return QueryResponse(response=response)

            if isinstance(response, dict):
                result = json.dumps(response, indent=2)
                return JSONResponse(
                    content=result, headers={"X-Affinity-Token": rag.affinity_token}
                )
            else:
                return JSONResponse(
                    content=str(response),
                    headers={"X-Affinity-Token": rag.affinity_token},
                )
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/query/stream", dependencies=[Depends(optional_api_key)])
    async def query_text_stream(
        request: QueryRequest,
        storage_account_url: str = Header(alias="Storage_Account_Url"),
        storage_container_name: str = Header(alias="Storage_Container_Name"),
        storage_token_expiry: str = Header(
            default=None, alias="Storage_Access_Token_Expiry"
        ),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        storage_access_token: str = Header(alias="Storage_Access_Token"),
        X_Affinity_Token: str = Header(None, alias="X-Affinity-Token"),
    ):
        """
        This endpoint performs a retrieval-augmented generation (RAG) query and streams the response.

        Args:
            request (QueryRequest): The request object containing the query parameters.
            optional_api_key (Optional[str], optional): An optional API key for authentication. Defaults to None.

        Returns:
            StreamingResponse: A streaming response containing the RAG query results.
        """
        if not ai_access_token or not storage_access_token:
            raise HTTPException(
                status_code=401,
                detail='Missing necessary authentication header: "Azure-AI-Access-Token" or "Storage_Access_Token"',
            )
        ai_access_token = extract_token_value(ai_access_token, "Azure-AI-Access-Token")
        storage_access_token = extract_token_value(
            storage_access_token, "Storage_Access_Token"
        )
        try:
            param = request.to_query_params(True)
            rag = initialize_rag_with_header(
                rag_instance_manager,
                storage_account_url,
                storage_container_name,
                X_Affinity_Token,
                storage_access_token,
                storage_token_expiry,
            )
            await wait_for_storage_initialization(
                rag,
                get_lightrag_token_credential(
                    storage_access_token, storage_token_expiry
                ),
            )
            response = await rag.aquery(request.query, param=param)

            from fastapi.responses import StreamingResponse

            async def stream_generator():
                if isinstance(response, str):
                    # If it's a string, send it all at once
                    yield f"{json.dumps({'response': response})}\n"
                else:
                    # If it's an async generator, send chunks one by one
                    try:
                        async for chunk in response:
                            if chunk:  # Only send non-empty content
                                yield f"{json.dumps({'response': chunk})}\n"
                    except Exception as e:
                        logging.error(f"Streaming error: {str(e)}")
                        yield f"{json.dumps({'error': str(e)})}\n"

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "application/x-ndjson",
                    "X-Affinity-Token": rag.affinity_token,
                    "X-Accel-Buffering": "no",  # Ensure proper handling of streaming response when proxied by Nginx
                },
            )
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    return router
