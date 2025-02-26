from pydantic import BaseModel, Field
from typing import Optional

class BaseRequest(BaseModel):
    llm_model: Optional[str] = Field(
        default=None,
        description="Model name for the LLM.",
    )

    llm_endpoint: Optional[str] = Field(
        default=None,
        description="Endpoint for the LLM.",
    )

    llm_api_version: Optional[str] = Field(
        default=None,
        description="API version for the LLM.",
    )

    embedding_model: Optional[str] = Field(
        default=None,
        description="Model name for the embedding.",
    )

    embedding_api_version: Optional[str] = Field(
        default=None,
        description="API version for the embedding.",
    )

    embedding_dimension: Optional[int] = Field(
        default=None,
        description="Dimension of the embedding.",
    )

    max_embedding_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens for the embedding.",
    )

    embedding_endpoint: Optional[str] = Field(
        default=None,
        description="Endpoint for the embedding.",
    )

    storage_connection_string: Optional[str] = Field(
        default=None,
        description="Connection string for the storage.",
    )

    storage_container_name: Optional[str] = Field(
        default=None,
        description="Container name for the storage.",
    )