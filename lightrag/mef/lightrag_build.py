import json
import logging
import logging.config
import os
import traceback
from typing import List, Dict, Any
from lightrag import LightRAG
from lightrag.api.utils_api import extract_token_value
from lightrag.rag_instance_manager import RAGInstanceManager


class LightragBuild:

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        logging_config_path = os.path.join(current_dir, "..", "logging_config.json")
        with open(logging_config_path, "r") as f:
            logging_configs = json.load(f)
            logging.config.dictConfig(logging_configs)

    async def _pipeline_index_texts(
        self,
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

    async def insert_text(
        self,
        input_text: str,
        source_id: str,
        ai_access_token: str,
        rag_configs: Dict[str, Any],
        db_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Insert text into the RAG system.

        This endpoint allows you to insert text data into the RAG system for later retrieval
        and use in generating responses.
        """
        rag: LightRAG | None = None
        response = {}
        try:
            rag_instance_manager = RAGInstanceManager(args=rag_configs)
            ai_access_token = extract_token_value(
                ai_access_token, "Azure-AI-Access-Token"
            )
            storage_access_token = extract_token_value(
                db_params["db_access_token"], "DB_Access_Token"
            )
            rag = await rag_instance_manager.get_rag_instance(
                db_url=db_params["db_url"],
                db_name=db_params["db_name"],
                db_user_name=db_params["db_user_name"],
                db_access_token=storage_access_token,
            )
            await self._pipeline_index_texts(
                rag, [input_text], [source_id], ai_access_token
            )
            response = {"response": "Text successfully received and indexed."}
        except Exception as e:
            response = {
                "error": f"error: {str(e)}\n, traceback: {str(traceback.format_exc())}"
            }
        finally:
            if rag:
                await rag.finalize_storages()
            return response
