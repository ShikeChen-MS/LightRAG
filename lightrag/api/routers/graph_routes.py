"""
This module contains all graph-related routes for the LightRAG API.
"""

import json
import logging
import traceback
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from ..utils_api import (
    get_api_key_dependency,
    extract_token_value,
)
from ... import LightRAG

router = APIRouter(tags=["graph"])


def create_graph_routes(rag_instance_manager, api_key: Optional[str] = None):
    optional_api_key = get_api_key_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(optional_api_key)])
    async def get_graph_labels(
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ):
        """Get all graph labels"""
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
            res = await rag.get_graph_labels()
            return JSONResponse(content=json.dumps(res))
        except Exception as e:
            logging.error(f"Error /graph/label/list: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    @router.get("/graphs", dependencies=[Depends(optional_api_key)])
    async def get_knowledge_graph(
        label: str,
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
        max_depth: int = 3,
    ):
        """Get knowledge graph for a specific label"""
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
            res = await rag.get_knowledge_graph(node_label=label, max_depth=max_depth)
            return JSONResponse(content=res.model_dump())
        except Exception as e:
            logging.error(f"Error /graphs: {str(e)}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if rag:
                await rag.finalize_storages()

    return router
