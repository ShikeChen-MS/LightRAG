from pathlib import Path
from typing import List

from azure.storage.blob import BlobServiceClient, BlobLeaseClient
from .az_token_credential import LightRagTokenCredential


class DocumentManager:
    def __init__(
        self,
        input_dir: str,
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
        ),
    ):
        self.input_dir = input_dir
        self.supported_extensions = supported_extensions
        self.indexed_files = set()

    def scan_directory_for_new_files(
        self,
        storage_account_url: str,
        storage_container_name: str,
        access_token: LightRagTokenCredential,
    ) -> List[Path]:
        """Scan input directory for new files"""
        new_files = []
        blob_client = BlobServiceClient(
            account_url=storage_account_url, credential=access_token
        )
        container_client = blob_client.get_container_client(storage_container_name)
        container_client.get_container_properties()  # this is to check if the container exists and authentication is valid
        lease: BlobLeaseClient = container_client.acquire_lease()
        blob_list = container_client.list_blob_names()
        for blob_name in blob_list:
            if self.is_supported_file(blob_name) in self.supported_extensions:
                route = f"{self.input_dir}/{blob_name}"
                if route not in self.indexed_files:
                    new_files.append(route)
        lease.release()
        return new_files

    # def scan_directory(self) -> List[Path]:
    #     new_files = []
    #     for ext in self.supported_extensions:
    #         for file_path in self.input_dir.rglob(f"*{ext}"):
    #             new_files.append(file_path)
    #     return new_files

    def mark_as_indexed(self, file_path: Path):
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)
