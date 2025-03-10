"""
Utility functions for the LightRAG API.
"""

import asyncio
import os
import argparse
from typing import Optional
from azure.storage.blob import BlobLeaseClient, BlobClient, ContainerClient
from fastapi import HTTPException, Security
from dotenv import load_dotenv
from fastapi.security import APIKeyHeader
from .. import LightRAG
from ..az_token_credential import LightRagTokenCredential
from ..base import InitializeStatus
import logging

# Load environment variables
load_dotenv(override=True)


def get_api_key_dependency(api_key: Optional[str]):
    """
    Create an API key dependency for route protection.

    Args:
        api_key (Optional[str]): The API key to validate against.
                                If None, no authentication is required.

    Returns:
        Callable: A dependency function that validates the API key.
    """
    if not api_key:
        # If no API key is configured, return a dummy dependency that always succeeds
        async def no_auth():
            return None

        return no_auth

    # If API key is configured, use proper authentication
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def api_key_auth(
        api_key_header_value: Optional[str] = Security(api_key_header),
    ):
        if not api_key_header_value:
            raise HTTPException(
                status_code=403, detail="API Key required"
            )
        if api_key_header_value != api_key:
            raise HTTPException(
                status_code=403, detail="Invalid API Key"
            )
        return api_key_header_value

    return api_key_auth


class DefaultRAGStorageConfig:
    KV_STORAGE = "JsonKVStorage"
    VECTOR_STORAGE = "NanoVectorDBStorage"
    GRAPH_STORAGE = "NetworkXStorage"
    DOC_STATUS_STORAGE = "JsonDocStatusStorage"


def get_default_host(binding_type: str) -> str:
    default_hosts = {
        "ollama": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
        "lollms": os.getenv("LLM_BINDING_HOST", "http://localhost:9600"),
        "azure_openai": os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.openai.com/v1"),
        "openai": os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
    }
    return default_hosts.get(
        binding_type, os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    )  # fallback to ollama if unknown


def get_env_value(env_key: str, default: any, value_type: type = str) -> any:
    """
    Get value from environment variable with type conversion

    Args:
        env_key (str): Environment variable key
        default (any): Default value if env variable is not set
        value_type (type): Type to convert the value to

    Returns:
        any: Converted value from environment or default
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    if value_type is bool:
        return value.lower() in ("true", "1", "yes", "t", "on")
    try:
        return value_type(value)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="LightRAG FastAPI Server with separate working and input directories"
    )

    # Server configuration
    parser.add_argument(
        "--host",
        default=get_env_value("HOST", "0.0.0.0"),
        help="Server host (default: from env or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_env_value("PORT", 9621, int),
        help="Server port (default: from env or 9621)",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default=get_env_value("WORKING_DIR", "lightrag"),
        help="Working (virtual) directory on Azure Storage Account (default: from env or lightrag)",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=get_env_value("INPUT_DIR", "input"),
        help="Input directory on Azure Storage Account (default: from env or input), this will be put within working directory",
    )

    def timeout_type(value):
        if value is None:
            return 150
        if value is None or value == "None":
            return None
        return int(value)

    parser.add_argument(
        "--timeout",
        default=get_env_value("TIMEOUT", None, timeout_type),
        type=timeout_type,
        help="Timeout in seconds (useful when using slow AI). Use None for infinite timeout",
    )

    # RAG configuration
    parser.add_argument(
        "--max-async",
        type=int,
        default=get_env_value("MAX_ASYNC", 4, int),
        help="Maximum async operations (default: from env or 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_env_value("MAX_TOKENS", 32768, int),
        help="Maximum token size (default: from env or 32768)",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        default=get_env_value("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: from env or INFO)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=get_env_value("VERBOSE", False, bool),
        help="Enable verbose debug output(only valid for DEBUG log-level)",
    )

    parser.add_argument(
        "--key",
        type=str,
        default=get_env_value("LIGHTRAG_API_KEY", None),
        help="API key for authentication. This protects lightrag server against unauthorized access",
    )

    # Optional https parameters
    parser.add_argument(
        "--ssl",
        action="store_true",
        default=get_env_value("SSL", False, bool),
        help="Enable HTTPS (default: from env or False)",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=get_env_value("SSL_CERTFILE", None),
        help="Path to SSL certificate file (required if --ssl is enabled)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=get_env_value("SSL_KEYFILE", None),
        help="Path to SSL private key file (required if --ssl is enabled)",
    )

    parser.add_argument(
        "--history-turns",
        type=int,
        default=get_env_value("HISTORY_TURNS", 3, int),
        help="Number of conversation history turns to include (default: from env or 3)",
    )

    # Search parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=get_env_value("TOP_K", 60, int),
        help="Number of most similar results to return (default: from env or 60)",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=get_env_value("COSINE_THRESHOLD", 0.2, float),
        help="Cosine similarity threshold (default: from env or 0.4)",
    )
    parser.add_argument(
        "--max-parallel-insert",
        type=int,
        default=get_env_value("MAX_PARALLEL_INSERT", 20, int),
        help="Maximum number of parallel insert operations (default: from env or 20)",
    )

    # Namespace
    parser.add_argument(
        "--namespace-prefix",
        type=str,
        default=get_env_value("NAMESPACE_PREFIX", ""),
        help="Prefix of the namespace",
    )

    args = parser.parse_args()

    # Inject storage configuration from environment variables
    args.kv_storage = get_env_value(
        "LIGHTRAG_KV_STORAGE", DefaultRAGStorageConfig.KV_STORAGE
    )
    args.doc_status_storage = get_env_value(
        "LIGHTRAG_DOC_STATUS_STORAGE", DefaultRAGStorageConfig.DOC_STATUS_STORAGE
    )
    args.graph_storage = get_env_value(
        "LIGHTRAG_GRAPH_STORAGE", DefaultRAGStorageConfig.GRAPH_STORAGE
    )
    args.vector_storage = get_env_value(
        "LIGHTRAG_VECTOR_STORAGE", DefaultRAGStorageConfig.VECTOR_STORAGE
    )

    args.llm_binding_host = get_env_value("AZURE_OPENAI_ENDPOINT", None)
    args.embedding_binding_host = get_env_value("AZURE_OPENAI_EMBEDDING_ENDPOINT", None)
    args.llm_api_version = get_env_value("AZURE_OPENAI_API_VERSION", None)
    args.embedding_api_version = get_env_value(
        "AZURE_OPENAI_EMBEDDING_API_VERSION", None
    )
    # Inject model configuration
    args.llm_model = get_env_value("AZURE_OPENAI_MODEL_NAME", None)
    args.embedding_model = get_env_value("AZURE_OPENAI_EMDEDDING_MODEL_NAME", None)
    args.embedding_dim = get_env_value("EMBEDDING_DIM", 1024, int)
    args.max_embed_tokens = get_env_value("MAX_EMBED_TOKENS", 8192, int)

    # Inject chunk configuration
    args.chunk_size = get_env_value("CHUNK_SIZE", 1200, int)
    args.chunk_overlap_size = get_env_value("CHUNK_OVERLAP_SIZE", 100, int)

    return args


async def wait_for_storage_initialization(
    rag: LightRAG, token: LightRagTokenCredential
):
    if rag.initialize_status == InitializeStatus.INITIALIZED.value:
        return
    if rag.initialize_status == InitializeStatus.INITIALIZING.value:
        await asyncio.sleep(3)
    raise HTTPException(status_code=500, detail="Storage initialization failed...")


async def initialize_rag_with_header(
    rag_instance_manager,
    storage_account_url,
    storage_container_name,
    x_affinity_token,
    storage_access_token,
    storage_token_expiry,
):
    try:
        if x_affinity_token:
            rag = await rag_instance_manager.get_lightrag_by_affinity_token(x_affinity_token)
        else:
            rag = await rag_instance_manager.get_lightrag(
                storage_account_url=storage_account_url,
                storage_container_name=storage_container_name,
                access_token=get_lightrag_token_credential(
                    storage_access_token, storage_token_expiry
                ),
            )
        return rag
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_lightrag_token_credential(storage_access_token, storage_token_expiry):
    return LightRagTokenCredential(storage_access_token, storage_token_expiry)


async def try_get_container_lease(
    client: ContainerClient | BlobClient,
) -> BlobLeaseClient:
    retry_count = 0
    lease = None
    while lease is None and retry_count < 50:
        try:
            lease = client.acquire_lease()
        except Exception as e:
            retry_count += 1
            lease = None
            logging.warning(
                f"Failed to acquire lease, error detail: {str(e)}, retrying in 3 seconds..."
            )
            await asyncio.sleep(3)
    if lease is None:
        logging.error(f"Failed to acquire lease after 50 retries, error....")
        raise HTTPException(
            status_code=500, detail="Failed to acquire lease after 50 retries"
        )
    return lease


def extract_token_value(authorization: str, header_name: str) -> str:
    """Extract token value from authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail=f'Bearer Token in the Header named as "{header_name}" is required',
        )
    token_parts = authorization.split()
    if len(token_parts) != 2 or token_parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401, detail=f"Invalid header {header_name} format"
        )
    return token_parts[1]
