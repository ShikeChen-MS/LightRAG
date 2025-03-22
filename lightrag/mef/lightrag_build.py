import logging
import traceback
from typing import Dict
from dotenv import load_dotenv
from ai_models.model import Executable, Initializable
from lightrag.api.rag_instance_manager import RAGInstanceManager
from lightrag.api.routers.document_routes import pipeline_index_texts
from lightrag.api.utils_api import(
    extract_token_value,
    initialize_rag_with_header,
    parse_args,
    wait_for_storage_initialization,
    get_lightrag_token_credential
)
# TODO: following imports are a temporary workaround for long load time
# TODO: on graph db related module especially networkx and graspologic.
# TODO: This expected to be fix after migrate to Azure Database server for PostgreSQL.
# TODO: the workaround is to import the module here so LightRAG server will
# TODO: take longer to start up but the initialization of the storage will be faster.
import lightrag.kg.json_doc_status_impl
import lightrag.kg.json_kv_impl
import lightrag.kg.nano_vector_db_impl
import lightrag.kg.networkx_impl
import lightrag.llm.azure_openai
import lightrag.lightrag
import lightrag.az_token_credential
import lightrag.base


class LightRAGBuilder:

    def __init__(self):
        """Inherited init."""
        Initializable.__init__(self)
        Executable.__init__(self)
        self.rag_instance_manager = None
        self.arguments = None

    def initialize(self, *args, **kwargs):
        """Load env variables and initialize RAGInstanceManager."""

        try:
            logging.info(f"RUNNING initialize with args: {args}")
            load_dotenv(override=True)
            self.arguments = parse_args()
            self.rag_instance_manager = RAGInstanceManager(args=self.arguments)

        except Exception as ex:
            logging.error(f"Error in LightRAGBuilder initialization: {str(ex)}")
            raise ex

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

            ai_access_token = extract_token_value(
                ai_access_token, "Azure-AI-Access-Token"
            )
            storage_access_token = extract_token_value(
                storage_access_token, "Storage_Access_Token"
            )
            rag = await initialize_rag_with_header(
                self.rag_instance_manager,
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