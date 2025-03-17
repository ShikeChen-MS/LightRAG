"""
LightRAG FastAPI Server
"""

import json
import traceback

from fastapi import (
    FastAPI,
    Depends,
    Header,
)
import uvicorn

from ..lightrag import LightRAG
from ..rag_instance_manager import RAGInstanceManager
import os
import logging.config
import configparser
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from .utils_api import (
    get_api_key_dependency,
    parse_args,
    extract_token_value,
)
from . import __api_version__
import logging
from .routers.document_routes import create_document_routes
from .routers.query_routes import create_query_routes
from .routers.graph_routes import create_graph_routes

# Load environment variables
try:
    load_dotenv(override=True)
except Exception as e:
    logging.warning(f"Failed to load .env file: {e}")

# Initialize config parser
config = configparser.ConfigParser()
config.read("config.ini")


def create_app(args, rag_instance_manager):
    # Set global top_k
    global global_top_k
    global_top_k = args["top_k"]  # save top_k from args

    logging.info("LightRAG server starting...")
    logging.info("Output all arguments collected...")
    logging.info("################################################################")
    for arg in args:
        logging.info(f"{str(arg)}: {args[arg]}")
    logging.info("################################################################")
    # Add SSL validation
    if args["ssl"]:
        if not args["ssl_certfile"] or not args["ssl_keyfile"]:
            raise Exception(
                "SSL certificate and key files must be provided when SSL is enabled"
            )
        if not os.path.exists(args["ssl_certfile"]):
            raise Exception(f"SSL certificate file not found: {args["ssl_certfile"]}")
        if not os.path.exists(args["ssl_keyfile"]):
            raise Exception(f"SSL key file not found: {args["ssl_keyfile"]}")

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args["key"]

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
    app.include_router(create_query_routes(rag_instance_manager, api_key, args["top_k"]))
    app.include_router(create_graph_routes(rag_instance_manager, api_key))

    @app.get("/logs", dependencies=[Depends(optional_api_key)])
    async def get_logs(
        storage_account_url: str = Header(alias="Storage_Account_Url"),
        storage_container_name: str = Header(alias="Storage_Container_Name"),
        storage_token_expiry: str = Header(
            default=None, alias="Storage_Access_Token_Expiry"
        ),
        storage_access_token: str = Header(alias="Storage_Access_Token"),
    ):
        current_dir = os.path.dirname(__file__)
        logging_config_path = os.path.join(current_dir, "logging_config.json")
        with open(logging_config_path, "r") as f:
            logging_configs = json.load(f)
            logfile_name = logging_configs["handlers"]["file"]["filename"]
        if os.path.isfile(logfile_name):
            return FileResponse(logfile_name)
        return JSONResponse(status_code=404, content="Log file not found.")

    @app.delete("/logs", dependencies=[Depends(optional_api_key)])
    async def delete_logs(
        storage_account_url: str = Header(alias="Storage_Account_Url"),
        storage_container_name: str = Header(alias="Storage_Container_Name"),
        storage_token_expiry: str = Header(
            default=None, alias="Storage_Access_Token_Expiry"
        ),
        storage_access_token: str = Header(alias="Storage_Access_Token"),
    ):
        current_dir = os.path.dirname(__file__)
        logging_config_path = os.path.join(current_dir, "logging_config.json")
        with open(logging_config_path, "r") as f:
            logging_configs = json.load(f)
            logfile_name = logging_configs["handlers"]["file"]["filename"]
        if os.path.isfile(logfile_name):
            os.remove(logfile_name)
            return JSONResponse(status_code=200, content="Log file deleted.")
        return JSONResponse(status_code=404, content="Log file not found.")

    @app.post("/health", dependencies=[Depends(optional_api_key)])
    async def get_status(
        db_url: str = Header(alias="DB_Url"),
        db_name: str = Header(alias="DB_Name"),
        db_user_name: str = Header(alias="DB_User_Name"),
        ai_access_token: str = Header(alias="Azure-AI-Access-Token"),
        db_access_token: str = Header(alias="DB_Access_Token"),
    ):
        """Get current system status"""
        result = {}
        # Collect all non-None arguments
        result["Status"] = "Healthy"
        for arg in args:
            if args[arg] is not None:
                name = arg.replace("_", " ")
                name = name.title()
                result[name] = args[arg]
        affinity_token = ""
        # initialize rag instance
        # send an example prompt to the model to check if it is working
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
        except Exception as e:
            result["Status"] = "Unhealthy"
            error = str(e) + str(traceback.format_exc())
            result["Error"] = error
        finally:
            if rag:
                await rag.finalize_storages()
        return JSONResponse(content=result)

    return app


def main():
    args = vars(parse_args())
    rag_instance_manager = RAGInstanceManager(args=args)
    current_dir = os.path.dirname(__file__)
    logging_config_path = os.path.join(current_dir, "..", "logging_config.json")
    with open(logging_config_path, "r") as f:
        logging_configs = json.load(f)
        # Configure uvicorn logging
        logging.config.dictConfig(logging_configs)
    app = create_app(args, rag_instance_manager)
    uvicorn_config = {
        "app": app,
        "host": args["host"],
        "port": args["port"],
        "log_config": None,  # Disable default config
    }
    if args["ssl"]:
        uvicorn_config.update(
            {
                "ssl_certfile": args["ssl_certfile"],
                "ssl_keyfile": args["ssl_keyfile"],
            }
        )
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
