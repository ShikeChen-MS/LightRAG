import configparser
import os
import uvicorn
import lightrag.api.server_util as server_util
import lightrag.api.server_app as server_app
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

rag_storage_config = server_util.RAGStorageConfig()

# read config.ini
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")
# Redis config
redis_uri = config.get("redis", "uri", fallback=None)
if redis_uri:
    os.environ["REDIS_URI"] = redis_uri
    rag_storage_config.KV_STORAGE = "RedisKVStorage"
    rag_storage_config.DOC_STATUS_STORAGE = "RedisKVStorage"

# Neo4j config
neo4j_uri = config.get("neo4j", "uri", fallback=None)
neo4j_username = config.get("neo4j", "username", fallback=None)
neo4j_password = config.get("neo4j", "password", fallback=None)
if neo4j_uri:
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USERNAME"] = neo4j_username
    os.environ["NEO4J_PASSWORD"] = neo4j_password
    rag_storage_config.GRAPH_STORAGE = "Neo4JStorage"

# Milvus config
milvus_uri = config.get("milvus", "uri", fallback=None)
milvus_user = config.get("milvus", "user", fallback=None)
milvus_password = config.get("milvus", "password", fallback=None)
milvus_db_name = config.get("milvus", "db_name", fallback=None)
if milvus_uri:
    os.environ["MILVUS_URI"] = milvus_uri
    os.environ["MILVUS_USER"] = milvus_user
    os.environ["MILVUS_PASSWORD"] = milvus_password
    os.environ["MILVUS_DB_NAME"] = milvus_db_name
    rag_storage_config.VECTOR_STORAGE = "MilvusVectorDBStorge"

# MongoDB config
mongo_uri = config.get("mongodb", "uri", fallback=None)
mongo_database = config.get("mongodb", "LightRAG", fallback=None)
if mongo_uri:
    os.environ["MONGO_URI"] = mongo_uri
    os.environ["MONGO_DATABASE"] = mongo_database
    rag_storage_config.KV_STORAGE = "MongoKVStorage"
    rag_storage_config.DOC_STATUS_STORAGE = "MongoKVStorage"


# Main Function
def main():
    args = server_util.parse_args()
    app = server_app.create_app(args, rag_storage_config)
    server_util.display_splash_screen(args)
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
    }
    if args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": args.ssl_certfile,
                "ssl_keyfile": args.ssl_keyfile,
            }
        )
    uvicorn.run(**uvicorn_config)


# Main Entry Point for api server
if __name__ == "__main__":
    main()
