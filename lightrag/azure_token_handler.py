import msal
import jwt
import enum
from datetime import datetime, timedelta, timezone


class TokenScope(enum.Enum):
    CognitiveServices = "https://cognitiveservices.azure.com/.default offline_access"
    CosmosDB_Postgres = (
        "https://token.postgres.cosmos.azure.com/.default offline_access"
    )
    Storage = "https://storage.azure.com/.default offline_access"


class AzureToken:
    def __init__(self, token: str, refresh_token: str, token_scope: TokenScope):
        if not all([token, refresh_token]):
            raise ValueError("All tokens must be provided")
        self.token = token
        # refresh token is long-lived and will be refreshed every time
        # we get a new access token using it. So need to carry it
        # along with access token.
        self.refresh_token = refresh_token
        self.token_scope = token_scope

    def check_scope(self, token_scope: TokenScope):
        return self.token_scope.name == token_scope.name


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
            token.get("access_token"), token.get("refresh_token"), token_scope
        )

    @classmethod
    def refresh_token_if_near_expire(
        cls, azure_token: AzureToken, threshold_seconds=600
    ):
        res = cls.__is_token_near_expire(azure_token, threshold_seconds)
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
        return AzureToken(
            token.get("access_token"), token.get("refresh_token"), token_scope
        )

    @classmethod
    def __is_token_near_expire(cls, azure_token: AzureToken, threshold_seconds: int):
        # In theory since both token issued at same time, the remaining life should
        # be same, but for safety we check both and anyone of them near expire
        # we shall refresh both of them.
        decoded_token = jwt.decode(
            azure_token.token, options={"verify_signature": False}
        )
        # Extract the expiration time
        exp_timestamp = decoded_token.get("exp")
        if not exp_timestamp:
            raise ValueError("The access token does not contain an expiration time.")
        # Convert the expiration time to a datetime object
        exp_time = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
        # Calculate the time remaining until expiration
        current_time = datetime.now(tz=timezone.utc)
        time_remaining = exp_time - current_time
        # Check if the token is near expiration
        return time_remaining < timedelta(seconds=threshold_seconds)
