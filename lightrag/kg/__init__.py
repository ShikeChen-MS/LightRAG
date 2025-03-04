STORAGE_IMPLEMENTATIONS = {
    "KV_STORAGE": {
        "implementations": [
            "JsonKVStorage",
            "PGKVStorage"
        ],
        "required_methods": ["get_by_id", "upsert"],
    },
    "GRAPH_STORAGE": {
        "implementations": [
            "NetworkXStorage",
            "PGGraphStorage",
        ],
        "required_methods": ["upsert_node", "upsert_edge"],
    },
    "VECTOR_STORAGE": {
        "implementations": [
            "NanoVectorDBStorage",
            "PGVectorStorage",
        ],
        "required_methods": ["query", "upsert"],
    },
    "DOC_STATUS_STORAGE": {
        "implementations": [
            "JsonDocStatusStorage",
            "PGDocStatusStorage",
        ],
        "required_methods": ["get_docs_by_status"],
    },
}

# Storage implementation environment variable without default value
STORAGE_ENV_REQUIREMENTS: dict[str, list[str]] = {
    # KV Storage Implementations
    "JsonKVStorage": [],
    "PGKVStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    # Graph Storage Implementations
    "NetworkXStorage": [],
    "PGGraphStorage": [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
    ],
    # Vector Storage Implementations
    "NanoVectorDBStorage": [],
    "PGVectorStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
    # Document Status Storage Implementations
    "JsonDocStatusStorage": [],
    "PGDocStatusStorage": ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DATABASE"],
}

# Storage implementation module mapping
STORAGES = {
    "NetworkXStorage": ".kg.networkx_impl",
    "JsonKVStorage": ".kg.json_kv_impl",
    "NanoVectorDBStorage": ".kg.nano_vector_db_impl",
    "JsonDocStatusStorage": ".kg.json_doc_status_impl",
    "PGKVStorage": ".kg.postgres_impl",
    "PGVectorStorage": ".kg.postgres_impl",
    "PGGraphStorage": ".kg.postgres_impl",
    "PGDocStatusStorage": ".kg.postgres_impl",
}


def verify_storage_implementation(storage_type: str, storage_name: str) -> None:
    """Verify if storage implementation is compatible with specified storage type

    Args:
        storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
        storage_name: Storage implementation name

    Raises:
        ValueError: If storage implementation is incompatible or missing required methods
    """
    if storage_type not in STORAGE_IMPLEMENTATIONS:
        raise ValueError(f"Unknown storage type: {storage_type}")

    storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
    if storage_name not in storage_info["implementations"]:
        raise ValueError(
            f"Storage implementation '{storage_name}' is not compatible with {storage_type}. "
            f"Compatible implementations are: {', '.join(storage_info['implementations'])}"
        )
