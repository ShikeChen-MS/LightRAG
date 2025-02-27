import azure.storage.blob as blob
import lightrag.azure_token_handler as token_handler

class StorageManager:

    def __init__(
            self,
            name: str,
            endpoint: str,
            storage_prefix: str,
            supported_extensions: tuple = (
                    ".txt",
                    ".md",
                    ".pdf",
                    ".docx",
                    ".pptx",
                    ".xlsx",
                    ".rtf",  # Rich Text Format
                    ".odt",  # OpenDocument Text
                    ".tex",  # LaTeX
                    ".epub",  # Electronic Publication
                    ".html",  # HyperText Markup Language
                    ".htm",  # HyperText Markup Language
                    ".csv",  # Comma-Separated Values
                    ".json",  # JavaScript Object Notation
                    ".xml",  # eXtensible Markup Language
                    ".yaml",  # YAML Ain't Markup Language
                    ".yml",  # YAML
                    ".log",  # Log files
                    ".conf",  # Configuration files
                    ".ini",  # Initialization files
                    ".properties",  # Java properties files
                    ".sql",  # SQL scripts
                    ".bat",  # Batch files
                    ".sh",  # Shell scripts
                    ".c",  # C source code
                    ".cpp",  # C++ source code
                    ".py",  # Python source code
                    ".java",  # Java source code
                    ".js",  # JavaScript source code
                    ".ts",  # TypeScript source code
                    ".swift",  # Swift source code
                    ".go",  # Go source code
                    ".rb",  # Ruby source code
                    ".php",  # PHP source code
                    ".css",  # Cascading Style Sheets
                    ".scss",  # Sassy CSS
                    ".less",  # LESS CSS
            )
    ):
        self.__input_container_name = f"{storage_prefix}-inputs"
        self.__storage_container_name = f"{storage_prefix}-storage"
        self.__storage_account_name = name
        self.__storage_endpoint = endpoint
        self.__supported_extensions = supported_extensions
        self.indexed_files = set()

    def get_blob_service_client(self, access_token:token_handler.AzureToken) -> blob.BlobServiceClient:
        if not access_token.check_scope(token_handler.TokenScope.Storage):
            raise (ValueError
                   (f"Mismatch in access token scope, expected: {token_handler.TokenScope.Storage.value}, actual: {access_token.token_scope.value}"))
        access_token = token_handler.AzureTokenHandler.refresh_token_if_near_expire(access_token)
        return blob.BlobServiceClient(
            account_url=self.__storage_endpoint,
            credential=token_handler.OnBehalfTokenCredential(access_token),
        )

    def scan_container_for_new_files(self, access_token:token_handler.AzureToken):
        blob_service_client = self.get_blob_service_client(access_token)
        container_client = blob_service_client.get_container_client(self.__input_container_name)
        blobs = container_client.list_blobs()
        files = []
        for blob in blobs:
            entry = f"{self.__input_container_name}/{blob.name}"
            if blob.name.endswith(self.__supported_extensions) and (entry not in self.indexed_files):
                files.append(entry)
        return files







