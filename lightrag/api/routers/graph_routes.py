"""
This module contains all graph-related routes for the LightRAG API.
"""

import json
from typing import Optional
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from ..utils_api import (
    get_api_key_dependency,
    initialize_rag_with_header,
    wait_for_storage_initialization,
    get_lightrag_token_credential,
    extract_token_value,
)

router = APIRouter(tags=["graph"])


def create_graph_routes(rag_instance_manager, api_key: Optional[str] = None):
    optional_api_key = get_api_key_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(optional_api_key)])
    async def get_graph_labels(
        storage_account_url: str = Header(None, alias="Storage_Account_Url"),
        storage_container_name: str = Header(None, alias="Storage_Container_Name"),
        storage_token_expiry: str = Header(None, alias="Storage_Access_Token_Expiry"),
        storage_access_token: str = Header(None, alias="Storage_Access_Token"),
        X_Affinity_Token: str = Header(None, alias="X-Affinity-Token"),
    ):
        """Get all graph labels"""
        if not storage_access_token:
            raise HTTPException(
                status_code=401,
                detail='Missing necessary authentication header: "Storage_Access_Token"',
            )
        storage_access_token = extract_token_value(
            storage_access_token, "Storage_Access_Token"
        )
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
            get_lightrag_token_credential(storage_access_token, storage_token_expiry),
        )
        res = await rag.get_graph_labels()
        return JSONResponse(
            content=json.dumps(res), headers={"X-Affinity-Token": rag.affinity_token}
        )

    @router.get("/graphs", dependencies=[Depends(optional_api_key)])
    async def get_knowledge_graph(
        label: str,
        storage_account_url: str = Header(None, alias="Storage_Account_Url"),
        storage_container_name: str = Header(None, alias="Storage_Container_Name"),
        storage_token_expiry: str = Header(None, alias="Storage_Access_Token_Expiry"),
        storage_access_token: str = Header(None, alias="Storage_Access_Token"),
        X_Affinity_Token: str = Header(None, alias="X-Affinity-Token"),
        max_depth: int = 3,
    ):
        """Get knowledge graph for a specific label"""
        if not not storage_access_token:
            raise HTTPException(
                status_code=401,
                detail='Missing necessary authentication header: "Storage_Access_Token"',
            )
        storage_access_token = extract_token_value(
            storage_access_token, "Storage_Access_Token"
        )
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
            get_lightrag_token_credential(storage_access_token, storage_token_expiry),
        )
        res = await rag.get_knowledge_graph(node_label=label, max_depth=max_depth)
        return JSONResponse(
            content=json.dumps(res), headers={"X-Affinity-Token": rag.affinity_token}
        )

    return router
