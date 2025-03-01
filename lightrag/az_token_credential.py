from azure.core.credentials import TokenCredential, AccessToken
from typing import Optional, Any


class LightRagTokenCredential(TokenCredential):

    def __init__(self, access_token: str, expires_on: int):
        self.access_token = access_token
        self.expire = expires_on

    def get_token(
        self,
        *scopes: str,
        claims: Optional[str] = None,
        tenant_id: Optional[str] = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AccessToken:
        return AccessToken(token=self.access_token, expires_on=self.expire)
