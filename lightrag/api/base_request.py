from pydantic import BaseModel, Field


class BaseRequest(BaseModel):

    storage_account_url: str = Field(
        default=None,
        description="URL for the storage account.",
    )

    storage_container_name: str = Field(
        default=None,
        description="Container name for the storage.",
    )

    ai_token_expiry: int = Field(
        default=0,
        description="Expiry time for Azure AI token in Unix time.",
    )

    storage_token_expiry: int = Field(
        default=0,
        description="Expiry time for Azure storage token in Unix time.",
    )
