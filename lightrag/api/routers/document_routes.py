"""
This module contains all document-related routes for the LightRAG API.
"""

import asyncio
import logging
import traceback
from ...rag_instance_manager import RAGInstanceManager
from typing import Dict, List, Optional, Any
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    Header,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from ... import LightRAG
from ...base import DocProcessingStatus, DocStatus
from ..utils_api import (
    get_api_key_dependency,
    extract_token_value,
)


router = APIRouter(prefix="/documents", tags=["documents"])

# Lock for thread-safe operations
progress_lock = asyncio.Lock()

# Temporary file prefix
temp_prefix = "__tmp__"


class InsertTextRequest(BaseModel):
    text: str = Field(
        min_length=1,
        description="The text to insert",
    )
    source_id: str = Field(
        min_length=1,
        description="The source id of the text to insert",
    )

    @field_validator("text", mode="after")
    @classmethod
    def strip_after(cls, text: str) -> str:
        return text.strip()


class InsertTextsRequest(BaseModel):
    texts: list[str] = Field(
        min_length=1,
        description="The texts to insert",
    )
    source_ids: list[str] = Field(
        min_length=1,
        description="The source ids of the texts to insert",
    )

    @field_validator("texts", mode="after")
    @classmethod
    def strip_after(cls, texts: list[str]) -> list[str]:
        return [text.strip() for text in texts]


class InsertResponse(BaseModel):
    status: str = Field(description="Status of the operation")
    message: str = Field(description="Message describing the operation result")


class DocStatusResponse(BaseModel):
    @staticmethod
    def format_datetime(dt: Any) -> Optional[str]:
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat()

    """Response model for document status

    Attributes:
        id: Document identifier
        content_summary: Summary of document content
        content_length: Length of document content
        status: Current processing status
        created_at: Creation timestamp (ISO format string)
        updated_at: Last update timestamp (ISO format string)
        chunks_count: Number of chunks (optional)
        error: Error message if any (optional)
        metadata: Additional metadata (optional)
    """

    id: str
    content_summary: str
    content_length: int
    status: DocStatus
    created_at: str
    updated_at: str
    chunks_count: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class DocsStatusesResponse(BaseModel):
    statuses: Dict[DocStatus, List[DocStatusResponse]] = {}


async def pipeline_enqueue_file(
    rag: LightRAG,
    file_name: str,
    content: str,
) -> bool:
    """Add a file to the queue for processing"""
    lease = None
    blob_lease = None
    try:
        # Insert into the RAG queue
        if content:
            await rag.apipeline_enqueue_documents([file_name], content)
            logging.info(f"Successfully fetched and enqueued file: {file_name}")
            return True
        else:
            logging.error(f"No content could be extracted from file: {file_name}")
            return False

    except Exception as e:
        logging.error(f"Error processing or enqueueing file {file_name}: {str(e)}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


async def pipeline_index_file(
    rag: LightRAG, file_name: str, content: str, ai_access_token: str
):
    """Index a file"""
    try:
        if await pipeline_enqueue_file(rag, file_name, content):
            await rag.apipeline_process_enqueue_documents(ai_access_token)

    except Exception as e:
        logging.error(f"Error indexing file {file_name}: {str(e)}")
        logging.error(traceback.format_exc())


async def pipeline_index_texts(
    rag: LightRAG,
    texts: List[str],
    source_ids: List[str],
    ai_access_token: str,
):
    """Index a list of texts"""
    if not texts:
        return
    await rag.apipeline_enqueue_documents(source_ids, texts)
    await rag.apipeline_process_enqueue_documents(ai_access_token)


def create_document_routes(
    rag_instance_manager: RAGInstanceManager, api_key: Optional[str] = None
):
    optional_api_key = get_api_key_dependency(api_key)

    @router.post(
        "/text", response_model=InsertResponse, dependencies=[Depends(optional_api_key)]
    )
    async def insert_text(
        request: InsertTextRequest,
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ) -> JSONResponse | None:
        """
        Insert text into the RAG system.

        This endpoint allows you to insert text data into the RAG system for later retrieval
        and use in generating responses.
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

            await pipeline_index_texts(
                rag, [request.text], [request.source_id], ai_access_token
            )
            response = JSONResponse(
                content={
                    "status": "success",
                    "message": "Text successfully received and indexed.",
                }
            )
            return response
        except Exception as e:
            logging.error(f"Error /documents/text: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    @router.post(
        "/texts",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_texts(
        request: InsertTextsRequest,
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ) -> JSONResponse | None:
        """
        Insert multiple texts into the RAG system.

        This endpoint allows you to insert multiple text entries into the RAG system
        in a single request.
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
            await pipeline_index_texts(
                rag, request.texts, request.source_ids, ai_access_token
            )
            response = JSONResponse(
                content={
                    "status": "success",
                    "message": "Text successfully received and indexed",
                }
            )
            return response
        except Exception as e:
            logging.error(f"Error /documents/text: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    @router.post(
        "/file", response_model=InsertResponse, dependencies=[Depends(optional_api_key)]
    )
    async def insert_file(
        file: UploadFile = File(...),
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ) -> JSONResponse | None:
        """
        Insert a file directly into the RAG system.

        This endpoint accepts a file upload and processes it for inclusion in the RAG system.
        The file is saved temporarily and processed in the background.
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
            if not file.filename.lower().endswith("txt"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Only .txt files are supported.",
                )
            filename = file.filename
            if " " in file.filename:
                filename = filename.replace(" ", "_")
                logging.info(
                    f"Renamed file {file.filename} to {filename} to remove spaces."
                )
            content = await file.read()
            utf8_content = content.decode("utf-8")
            await pipeline_index_file(
                rag,
                filename,
                utf8_content,
                ai_access_token,
            )
            response = JSONResponse(
                content={
                    "status": "success",
                    "message": f"File '{file.filename}' uploaded and indexed successfully.",
                }
            )
            return response
        except Exception as e:
            logging.error(f"Error /documents/file: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    @router.delete(
        "", response_model=InsertResponse, dependencies=[Depends(optional_api_key)]
    )
    async def clear_documents(
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ) -> JSONResponse:
        """
        Clear all documents from the RAG system.

        This endpoint deletes all text chunks, entities vector database, and relationships
        vector database, effectively clearing all documents from the RAG system.
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
            rag.text_chunks = []
            rag.entities_vdb = None
            rag.relationships_vdb = None
            await rag.clear_storages()
            response = JSONResponse(
                content={
                    "status": "success",
                    "message": "All databases has been deleted in remote storage, new empty databases has been created.",
                }
            )
            return response
        except Exception as e:
            logging.error(f"Error DELETE /documents: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    @router.get("", dependencies=[Depends(optional_api_key)])
    async def documents(
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ) -> JSONResponse:
        """
        Get the status of all documents in the system.

        This endpoint retrieves the current status of all documents, grouped by their
        processing status (PENDING, PROCESSING, PROCESSED, FAILED).
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
            statuses = (
                DocStatus.PENDING,
                DocStatus.PROCESSING,
                DocStatus.PROCESSED,
                DocStatus.FAILED,
            )
            tasks = [rag.get_docs_by_status(status) for status in statuses]
            results: List[Dict[str, DocProcessingStatus]] = await asyncio.gather(*tasks)

            response = DocsStatusesResponse()
            for idx, result in enumerate(results):
                status = statuses[idx]
                for doc_id, doc_status in result.items():
                    if status not in response.statuses:
                        response.statuses[status] = []
                    response.statuses[status].append(
                        DocStatusResponse(
                            id=doc_id,
                            content_summary=doc_status.content_summary,
                            content_length=doc_status.content_length,
                            status=doc_status.status,
                            created_at=DocStatusResponse.format_datetime(
                                doc_status.created_at
                            ),
                            updated_at=DocStatusResponse.format_datetime(
                                doc_status.updated_at
                            ),
                            chunks_count=doc_status.chunks_count,
                            error=doc_status.error,
                            metadata=doc_status.metadata,
                        )
                    )
            return JSONResponse(content=response.model_dump())
        except Exception as e:
            logging.error(f"Error GET /documents: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    return router
