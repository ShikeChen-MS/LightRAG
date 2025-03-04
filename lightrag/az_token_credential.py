from azure.core.credentials import TokenCredential, AccessToken
from typing import Optional, Any
from datetime import datetime, timezone, timedelta


class LightRagTokenCredential(TokenCredential):

    def __init__(self, access_token: str, expires_on: int):
        self.access_token = access_token
        if expires_on is None or expires_on == 0:
            expire_time = datetime.now(timezone.utc) + timedelta(hours=1)
            self.expires_on = int(expire_time.timestamp())
        else:
            self.expires_on = expires_on

    def get_token(
        self,
        *scopes: str,
        claims: Optional[str] = None,
        tenant_id: Optional[str] = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AccessToken:
        return AccessToken(token=self.access_token, expires_on=self.expires_on)
