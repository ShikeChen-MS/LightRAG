import azure.storage.blob as blob
import lightrag.azure_token_handler as token_handler
from adlfs import AzureBlobFileSystem

class StorageManager:
    __input_container_name = None
    __storage_container_name = None
    __storage_endpoint = None
    __storage_account_name = None

    def __init__(self, name: str, endpoint: str, storage_prefix: str):
        self.__input_container_name = f"{storage_prefix}-inputs"
        self.__storage_container_name = f"{storage_prefix}-storage"
        self.__storage_account_name = name
        self.__storage_endpoint = endpoint

    def get_blob_service_client(self, access_token:token_handler.AzureToken) -> blob.BlobServiceClient:
        if not access_token.check_scope(token_handler.TokenScope.Storage):
            raise (ValueError
                   (f"Mismatch in access token scope, expected: {token_handler.TokenScope.Storage.value}, actual: {access_token.token_scope.value}"))
        access_token = token_handler.AzureTokenHandler.refresh_token_if_near_expire(access_token)
        return blob.BlobServiceClient(
            account_url=self.__storage_endpoint,
            credential=token_handler.OnBehalfTokenCredential(access_token),
        )

    def get_azfilesystem(self, access_token:token_handler.AzureToken) -> AzureBlobFileSystem:
        if not access_token.check_scope(token_handler.TokenScope.Storage):
            raise (ValueError
                   (f"Mismatch in access token scope, expected: {token_handler.TokenScope.Storage.value}, actual: {access_token.token_scope.value}"))
        access_token = token_handler.AzureTokenHandler.refresh_token_if_near_expire(access_token)
        return AzureBlobFileSystem(
            account_name=self.__storage_account_name,
            credential=token_handler.OnBehalfTokenCredential(access_token)
        )



