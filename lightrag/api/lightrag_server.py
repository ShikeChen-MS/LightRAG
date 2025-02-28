"""
LightRAG FastAPI Server
"""

from fastapi import (
    FastAPI,
    Depends,
)
from fastapi.responses import FileResponse
from lightrag.api.rag_instance_manager import RAGInstanceManager
import asyncio
import threading
import os
from fastapi.staticfiles import StaticFiles
import logging
from typing import Dict
from pathlib import Path
import configparser
from lightrag.llm.azure_openai import (
    azure_openai_complete_if_cache,
    azure_openai_embed,
)
from ascii_colors import ASCIIColors
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from .utils_api import (
    get_api_key_dependency,
    parse_args,
    get_default_host,
    display_splash_screen,
)
from lightrag import LightRAG
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.api import __api_version__
from lightrag.utils import EmbeddingFunc
from lightrag.utils import logger
from .routers.document_routes import (
    DocumentManager,
    create_document_routes,
    run_scanning_process,
)
from .routers.query_routes import create_query_routes
from .routers.graph_routes import create_graph_routes
# Load environment variables
try:
    load_dotenv(override=True)
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")

# Initialize config parser
config = configparser.ConfigParser()
config.read("config.ini")

# Global configuration
global_top_k = 60  # default value

# Global progress tracker
scan_progress: Dict = {
    "is_scanning": False,
    "current_file": "",
    "indexed_count": 0,
    "total_files": 0,
    "progress": 0,
}

# Lock for thread-safe operations
progress_lock = threading.Lock()


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

    # Verify that bindings are correctly setup
    if args.llm_binding not in [
        "lollms",
        "ollama",
        "openai",
        "openai-ollama",
        "azure_openai",
    ]:
        raise Exception("llm binding not supported")

    if args.embedding_binding not in ["lollms", "ollama", "openai", "azure_openai"]:
        raise Exception("embedding binding not supported")

    # Set default hosts if not provided
    if args.llm_binding_host is None:
        args.llm_binding_host = get_default_host(args.llm_binding)

    if args.embedding_binding_host is None:
        args.embedding_binding_host = get_default_host(args.embedding_binding)

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

    # Initialize document manager
    doc_manager = DocumentManager(args.input_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        pass
        yield
        pass

    # Initialize FastAPI
    app = FastAPI(
        title="LightRAG API",
        description="API for querying text using LightRAG with separate storage and input directories"
        + "(With authentication)"
        if api_key
        else "",
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
    app.include_router(create_document_routes(rag_instance_manager, doc_manager, api_key))
    app.include_router(create_query_routes(rag_instance_manager, api_key, args.top_k))
    app.include_router(create_graph_routes(rag_instance_manager, api_key))

    @app.get("/health", dependencies=[Depends(optional_api_key)])
    async def get_status():
        """Get current system status"""
        return {
            "status": "healthy",
            "working_directory": str(args.working_dir),
            "input_directory": str(args.input_dir),
            "configuration": {
                # LLM configuration binding/host address (if applicable)/model (if applicable)
                "llm_binding": args.llm_binding,
                "llm_binding_host": args.llm_binding_host,
                "llm_model": args.llm_model,
                # embedding model configuration binding/host address (if applicable)/model (if applicable)
                "embedding_binding": args.embedding_binding,
                "embedding_binding_host": args.embedding_binding_host,
                "embedding_model": args.embedding_model,
                "max_tokens": args.max_tokens,
                "kv_storage": args.kv_storage,
                "doc_status_storage": args.doc_status_storage,
                "graph_storage": args.graph_storage,
                "vector_storage": args.vector_storage,
            },
        }

    # Webui mount webui/index.html
    static_dir = Path(__file__).parent / "webui"
    static_dir.mkdir(exist_ok=True)
    app.mount(
        "/webui",
        StaticFiles(directory=static_dir, html=True, check_dir=True),
        name="webui",
    )

    @app.get("/webui/")
    async def webui_root():
        return FileResponse(static_dir / "index.html")

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
