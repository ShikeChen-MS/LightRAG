import msal
import jwt
from datetime import datetime, timedelta, timezone


class AzureToken:
    def __init__(self, ai_token: str, cosmos_token: str, refresh_token: str):
        if not all([ai_token, cosmos_token, refresh_token]):
            raise ValueError("All tokens must be provided")
        self.ai_token = ai_token
        self.cosmos_token = cosmos_token
        self.refresh_token = refresh_token


class AzureTokenHandler:
    __client_id = None
    __client_secret = None
    __authority = (
        "https://login.microsoftonline.com/717ad1e8-d729-4966-ad8f-cf873ef57637"
    )
    __app_scope = [f"{__client_id}/.default offline_access"]
    __ai_scope = ["https://cognitiveservices.azure.com/.default offline_access"]
    __cosmos_scope = ["https://token.postgres.cosmos.azure.com/.default offline_access"]
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
    def get_azure_token_by_user_token(cls, access_token: str):
        app = cls.get_confidential_app()
        result = app.acquire_token_on_behalf_of(
            user_assertion=access_token, scopes=cls.__app_scope
        )
        if "error" in result:
            raise ValueError(f"Error acquiring token: {result['error_description']}")
        if "refresh_token" not in result:
            raise ValueError("Refresh token not found in the result")
        ai_token = app.acquire_token_by_refresh_token(
            result.get("refresh_token"), cls.__ai_scope
        )
        cosmos_token = app.acquire_token_by_refresh_token(
            ai_token.get("refresh_token"), cls.__cosmos_scope
        )
        return AzureToken(
            ai_token.get("access_token"),
            cosmos_token.get("access_token"),
            cosmos_token.get("refresh_token"),
        )

    @classmethod
    def refresh_token_if_near_expire(
        cls, azure_token: AzureToken, threshold_seconds=600
    ):
        res = cls.__is_token_near_expire(azure_token, threshold_seconds)
        if not res:
            return azure_token
        app = cls.get_confidential_app()
        if azure_token.refresh_token:
            ai_token = app.acquire_token_by_refresh_token(
                azure_token.refresh_token, cls.__ai_scope
            )
            cosmos_token = app.acquire_token_by_refresh_token(
                ai_token.get("refresh_token"), cls.__cosmos_scope
            )
            return AzureToken(
                ai_token.get("access_token"),
                cosmos_token.get("access_token"),
                cosmos_token.get("refresh_token"),
            )
        else:
            raise ValueError("Refresh token is missing")

    @classmethod
    def __is_token_near_expire(cls, azure_token: AzureToken, threshold_seconds: int):
        # In theory since both token issued at same time, the remaining life should
        # be same, but for safety we check both and anyone of them near expire
        # we shall refresh both of them.
        ai_decoded_token = jwt.decode(
            azure_token.ai_token, options={"verify_signature": False}
        )
        cosmos_decoded_token = jwt.decode(
            azure_token.cosmos_token, options={"verify_signature": False}
        )
        # Extract the expiration time
        ai_exp_timestamp = ai_decoded_token.get("exp")
        cosmos_exp_timestamp = cosmos_decoded_token.get("exp")
        if not ai_exp_timestamp:
            raise ValueError("The AI access token does not contain an expiration time.")
        if not cosmos_exp_timestamp:
            raise ValueError(
                "The Cosmos access token does not contain an expiration time."
            )
        # Convert the expiration time to a datetime object
        ai_exp_time = datetime.fromtimestamp(ai_exp_timestamp, tz=timezone.utc)
        cosmos_exp_time = datetime.fromtimestamp(cosmos_exp_timestamp, tz=timezone.utc)
        # Calculate the time remaining until expiration
        current_time = datetime.now(tz=timezone.utc)
        ai_time_remaining = ai_exp_time - current_time
        cosmos_time_remaining = cosmos_exp_time - current_time
        # Check if the token is near expiration
        is_near_expiry = ai_time_remaining < timedelta(seconds=threshold_seconds) or (
            cosmos_time_remaining < timedelta(seconds=threshold_seconds)
        )
        return is_near_expiry
