"""
LightRAG FastAPI Server
"""

from fastapi import (
    FastAPI,
    Depends,
    Header,
)
from lightrag.api.rag_instance_manager import RAGInstanceManager
import os
import logging
import configparser
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from .utils_api import (
    get_api_key_dependency,
    parse_args,
    display_splash_screen,
    initialize_rag_with_header,
    wait_for_storage_initialization,
    get_lightrag_token_credential,
    extract_token_value,
)
from . import __api_version__
from ..utils import logger
from .routers.document_routes import create_document_routes
from .routers.query_routes import create_query_routes
from .routers.graph_routes import create_graph_routes

# TODO: following imports are a temporary workaround for long load time
# TODO: on graph db related module especially networkx and graspologic.
# TODO: This expected to be fix after migrate to Azure Database server for PostgreSQL.
# TODO: the workaround is to import the module here so LightRAG server will
# TODO: take longer to start up but the initialization of the storage will be faster.
import lightrag.kg.json_doc_status_impl
import lightrag.kg.json_kv_impl
import lightrag.kg.nano_vector_db_impl
import lightrag.kg.networkx_impl


# Load environment variables
try:
    load_dotenv(override=True)
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")

# Initialize config parser
config = configparser.ConfigParser()
config.read("config.ini")


class AccessLogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        # Define paths to be filtered
        self.filtered_paths = ["/documents", "/health", "/webui/"]

    def filter(self, record):
        try:
            if not hasattr(record, "args") or not isinstance(record.args, tuple):
                return True
            if len(record.args) < 5:
                return True

            method = record.args[1]
            path = record.args[2]
            status = record.args[4]
            # print(f"Debug - Method: {method}, Path: {path}, Status: {status}")
            # print(f"Debug - Filtered paths: {self.filtered_paths}")

            if (
                method == "GET"
                and (status == 200 or status == 304)
                and path in self.filtered_paths
            ):
                return False

            return True

        except Exception:
            return True


def create_app(args, rag_instance_manager):
    # Set global top_k
    global global_top_k
    global_top_k = args.top_k  # save top_k from args

    # Initialize verbose debug setting
    from lightrag.utils import set_verbose_debug

    set_verbose_debug(args.verbose)

    # Add SSL validation
    if args.ssl:
        if not args.ssl_certfile or not args.ssl_keyfile:
            raise Exception(
                "SSL certificate and key files must be provided when SSL is enabled"
            )
        if not os.path.exists(args.ssl_certfile):
            raise Exception(f"SSL certificate file not found: {args.ssl_certfile}")
        if not os.path.exists(args.ssl_keyfile):
            raise Exception(f"SSL key file not found: {args.ssl_keyfile}")

    # Setup logging
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=getattr(logging, args.log_level)
    )

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        pass
        yield
        pass

    # Initialize FastAPI
    app = FastAPI(
        title="LightRAG API",
        description=(
            "API for querying text using LightRAG with separate storage and input directories"
            + "(With authentication)"
            if api_key
            else ""
        ),
        version=__api_version__,
        openapi_tags=[{"name": "api"}],
        lifespan=lifespan,
    )

    def get_cors_origins():
        """Get allowed origins from environment variable
        Returns a list of allowed origins, defaults to ["*"] if not set
        """
        origins_str = os.getenv("CORS_ORIGINS", "*")
        if origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in origins_str.split(",")]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create the optional API key dependency
    optional_api_key = get_api_key_dependency(api_key)

    # Add routes
    app.include_router(create_document_routes(rag_instance_manager, api_key))
    app.include_router(create_query_routes(rag_instance_manager, api_key, args.top_k))
    app.include_router(create_graph_routes(rag_instance_manager, api_key))

    @app.post("/health", dependencies=[Depends(optional_api_key)])
    async def get_status(
        storage_account_url: str = Header(None, alias="Storage_Account_Url"),
        storage_container_name: str = Header(None, alias="Storage_Container_Name"),
        storage_token_expiry: str = Header(None, alias="Storage_Access_Token_Expiry"),
        ai_access_token: str = Header(None, alias="Azure-AI-Access-Token"),
        storage_access_token: str = Header(None, alias="Storage_Access_Token"),
        X_Affinity_Token: str = Header(None, alias="X-Affinity-Token"),
    ):
        """Get current system status"""
        storage_access_token = extract_token_value(
            storage_access_token, "Storage_Access_Token"
        )
        ai_access_token = extract_token_value(ai_access_token, "Azure-AI-Access-Token")
        result = {}
        # Collect all non-None arguments
        result["Status"] = "Healthy"
        for arg in vars(args):
            if getattr(args, arg) is not None:
                name = arg.replace("_", " ")
                name = name.title()
                result[name] = getattr(args, arg)
        affinity_token = ""
        # initialize rag instance
        # send an example prompt to the model to check if it is working
        try:
            rag = initialize_rag_with_header(
                rag_instance_manager,
                storage_account_url,
                storage_container_name,
                X_Affinity_Token,
                storage_access_token,
                storage_token_expiry,
            )
            lightrag_token = get_lightrag_token_credential(
                storage_access_token, storage_token_expiry
            )
            await wait_for_storage_initialization(
                rag,
                lightrag_token,
            )
            result["LLM Test Prompt"] = (
                "Please tell me a trivial fact about the universe."
            )
            response = await rag.llm_model_func(
                access_token=ai_access_token, prompt=result["LLM Test Prompt"]
            )
            result["LLM Response"] = response
            result["Embedding Test Prompt"] = "Test text for embedding"
            response = await rag.embedding_func(
                aad_token=ai_access_token, texts=result["Embedding Test Prompt"]
            )
            result["Embedding Response Length"] = len(response[0])
            affinity_token = rag.affinity_token
        except Exception as e:
            result["Status"] = "Unhealthy"
            result["Error"] = str(e)
        return JSONResponse(
            content=result, headers={"X-Affinity-Token": affinity_token}
        )

    return app


def main():
    args = parse_args()
    rag_instance_manager = RAGInstanceManager(args=args)
    import uvicorn
    import logging.config

    # Configure uvicorn logging
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "uvicorn.access": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Add filter to uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(AccessLogFilter())

    app = create_app(args, rag_instance_manager)
    display_splash_screen(args)
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_config": None,  # Disable default config
    }
    if args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": args.ssl_certfile,
                "ssl_keyfile": args.ssl_keyfile,
            }
        )
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
