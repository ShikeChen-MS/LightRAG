from typing import Optional, Any
import msal
import enum
from datetime import datetime, timedelta, timezone
from azure.core.credentials import TokenCredential, AccessToken


class TokenScope(enum.Enum):
    CognitiveServices = "https://cognitiveservices.azure.com/.default offline_access"
    CosmosDB_Postgres = (
        "https://token.postgres.cosmos.azure.com/.default offline_access"
    )
    Storage = "https://storage.azure.com/.default offline_access"


class AzureToken:
    def __init__(self, token: str, refresh_token: str, expires_in_seconds: int, token_scope: TokenScope):
        if not all([token, refresh_token, expires_in_seconds]):
            raise ValueError("All tokens must be provided")
        self.token = token
        # refresh token is long-lived and will be refreshed every time
        # we get a new access token using it. So need to carry it
        # along with access token.
        self.refresh_token = refresh_token
        self.token_scope = token_scope
        self.token_expiry = datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in_seconds)

    def check_scope(self, token_scope: TokenScope):
        return self.token_scope.name == token_scope.name


class OnBehalfTokenCredential(TokenCredential):
    def __init__(self, access_token: AzureToken):
        self.access_token = access_token

    def get_token(
        self,
        *scopes: str,
        claims: Optional[str] = None,
        tenant_id: Optional[str] = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AccessToken:
        expiry_unix_time = int(self.access_token.token_expiry.timestamp())
        return AccessToken(token=self.access_token.token, expires_on=expiry_unix_time)


class AzureTokenHandler:
    __client_id = None
    # TODO: this is for development and debugging
    # TODO: Need to modify this implementation before production
    __client_secret = None
    __authority = None
    __confidential_app = None

    @classmethod
    def get_confidential_app(cls):
        if cls.__confidential_app is None:
            cls.__confidential_app = msal.ConfidentialClientApplication(
                client_id=cls.__client_id,
                client_credential=cls.__client_secret,
                authority=cls.__authority,
            )
        return cls.__confidential_app

    @classmethod
    def acquire_token_by_user_token(cls, access_token: str, token_scope: TokenScope):
        app = cls.get_confidential_app()
        # Acquire token on behalf of the user with scope to app itself
        # This will allow us to acquire token to any scope that
        # the app has access to with the refresh token we get
        result = app.acquire_token_on_behalf_of(
            user_assertion=access_token,
            scopes=[f"{cls.__client_id}/.default offline_access"],
        )
        if "error" in result:
            raise ValueError(f"Error acquiring token: {result['error_description']}")
        if "refresh_token" not in result:
            raise ValueError(
                "Refresh token not found in the result while acquiring token by user token"
            )
        token = app.acquire_token_by_refresh_token(
            result.get("refresh_token"), [token_scope.value]
        )
        return AzureToken(
            token.get("access_token"), token.get("refresh_token"), token.get("expires_in"), token_scope
        )

    @classmethod
    def refresh_token_if_near_expire(
            cls, azure_token: AzureToken, threshold_seconds=600
    ):
        # test if token is near expiry
        res = azure_token.token_expiry - datetime.now(tz=timezone.utc)
        if res.total_seconds() > threshold_seconds:
            return azure_token
        if not res:
            return azure_token
        app = cls.get_confidential_app()
        if not azure_token.refresh_token:
            raise ValueError("Refresh token is missing")
        try:
            token = app.acquire_token_by_refresh_token(
                azure_token.refresh_token, [azure_token.token_scope.value]
            )
            azure_token.token = token.get("access_token")
            azure_token.refresh_token = token.get("refresh_token")
            return azure_token
        except Exception as e:
            raise ValueError(f"Error acquiring token: {str(e)}")

    @classmethod
    def acquire_token_by_refresh_token(
            cls, refresh_token: str, token_scope: TokenScope
    ):
        app = cls.get_confidential_app()
        token = app.acquire_token_by_refresh_token(refresh_token, [token_scope.value])
        print(token)
        return AzureToken(
            token.get("access_token"), token.get("refresh_token"), token.get("expires_in"), token_scope
        )
