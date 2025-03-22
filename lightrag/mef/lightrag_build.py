import logging
import traceback
from typing import Dict
from dotenv import load_dotenv

from lightrag.api.rag_instance_manager import RAGInstanceManager
from lightrag.api.routers.document_routes import pipeline_index_texts
from lightrag.api.utils_api import extract_token_value, initialize_rag_with_header, parse_args, \
    wait_for_storage_initialization, get_lightrag_token_credential


class LightRAGBuilder:

    async def execute(
        self,
        text:str,
        source_id:str,
        storage_account_url: str,
        storage_container_name: str,
        storage_token_expiry: str,
        ai_access_token: str,
        storage_access_token: str,
    ) -> Dict[str, str]:
        """
        Insert text into the RAG system.

        This endpoint allows you to insert text data into the RAG system for later retrieval
        and use in generating responses.
        """
        try:
            load_dotenv(override=True)
            args = parse_args()
            rag_instance_manager = RAGInstanceManager(args=args)
            ai_access_token = extract_token_value(
                ai_access_token, "Azure-AI-Access-Token"
            )
            storage_access_token = extract_token_value(
                storage_access_token, "Storage_Access_Token"
            )
            rag = await initialize_rag_with_header(
                rag_instance_manager,
                storage_account_url,
                storage_container_name,
                None,
                storage_access_token,
                storage_token_expiry,
            )
            await wait_for_storage_initialization(
                rag,
                get_lightrag_token_credential(
                    storage_access_token, storage_token_expiry
                ),
            )
            await pipeline_index_texts(
                rag,
                [text],
                [source_id],
                ai_access_token,
                storage_account_url,
                storage_container_name,
                get_lightrag_token_credential(
                    storage_access_token, storage_token_expiry
                ),
            )
            response ={
                "status": "success",
                "message": "Text successfully received and indexed."
            }
            return response
        except Exception as e:
            logging.error(f"Error /documents/text: {str(e)}")
            logging.error(traceback.format_exc())
            return {"Error: Error occurred during insert text": f"detail: {str(e)}"}