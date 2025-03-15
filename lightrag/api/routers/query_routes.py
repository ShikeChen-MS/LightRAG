"""
This module contains all query-related routes for the LightRAG API.
"""

import json
import logging
import traceback
from typing import Any, Dict, List, Literal, Optional
from fastapi.responses import JSONResponse
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Header,
)
from fastapi.responses import StreamingResponse

from ... import LightRAG
from ...base import QueryParam
from ..utils_api import (
    get_api_key_dependency,
    extract_token_value,
)
from pydantic import BaseModel, Field, field_validator

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

    @router.delete("/query/cache", dependencies=[Depends(optional_api_key)])
    async def clear_query_cache(
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ) -> JSONResponse | None:
        rag: LightRAG | None = None
        try:
            storage_access_token = extract_token_value(
                db_access_token, "DB_Access_Token"
            )
            rag = await rag_instance_manager.get_rag_instance(
                db_url=db_url,
                db_name=db_name,
                db_user_name=db_user_name,
                db_access_token=storage_access_token,
            )
            await rag.llm_response_cache.clear()
            response = JSONResponse(
                content={
                    "status": "success",
                    "message": "LLM response cache cleared successfully.",
                }
            )
            return response
        except Exception as e:
            logging.error(f"Error /query/cache: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    @router.post(
        "/query", response_model=QueryResponse, dependencies=[Depends(optional_api_key)]
    )
    async def query_text(
        request: QueryRequest,
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ) -> JSONResponse | None:
        """
        Handle a POST request at the /query endpoint to process user queries using RAG capabilities.
        """
        rag: LightRAG | None = None
        try:
            ai_access_token = extract_token_value(
                ai_access_token, "Azure-AI-Access-Token"
            )
            storage_access_token = extract_token_value(
                db_access_token, "DB_Access_Token"
            )
            rag = await rag_instance_manager.get_rag_instance(
                db_url=db_url,
                db_name=db_name,
                db_user_name=db_user_name,
                db_access_token=storage_access_token,
            )
            param = request.to_query_params(False)
            response = await rag.aquery(
                request.query,
                ai_access_token,
                param=param,
            )
            # If response is a string (e.g. cache hit), return directly
            if isinstance(response, str):
                return JSONResponse(content={"response": response})
            if isinstance(response, dict):
                result = json.dumps(response, indent=2)
                return JSONResponse(content=result)
            else:
                return JSONResponse(content=str(response))
        except Exception as e:
            logging.error(f"Error /query: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    @router.post("/query/stream", dependencies=[Depends(optional_api_key)])
    async def query_text_stream(
        request: QueryRequest,
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ) -> StreamingResponse | None:
        """
        This endpoint performs a retrieval-augmented generation (RAG) query and streams the response.
        """
        rag: LightRAG | None = None
        try:
            ai_access_token = extract_token_value(
                ai_access_token, "Azure-AI-Access-Token"
            )
            storage_access_token = extract_token_value(
                db_access_token, "DB_Access_Token"
            )
            rag = await rag_instance_manager.get_rag_instance(
                db_url=db_url,
                db_name=db_name,
                db_user_name=db_user_name,
                db_access_token=storage_access_token,
            )
            param = request.to_query_params(False)
            response = await rag.aquery(
                request.query,
                ai_access_token,
                param=param,
            )

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
                    "X-Accel-Buffering": "no",  # Ensure proper handling of streaming response when proxied by Nginx
                },
            )
        except Exception as e:
            logging.error(f"Error /query/stream: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    return router
