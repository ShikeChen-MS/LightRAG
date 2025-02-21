import threading
from cachetools import TTLCache

class AzureOpenaiClient:
    _instance = None
    _lock = threading.Lock()

#    1. Singleton pattern to ensure only one instance of Azure_Openai_Client exists.
    def __new__(cls, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(AzureOpenaiClient, cls).__new__(cls)
                    cls._instance._init_kwargs = kwargs
        return cls._instance

    def __init__(self, **kwargs):
        if not hasattr(self, 'initialized'):
            self.initialized = True
        # by default, allow the cache to grow to 1000 items
        # per documentation, the LRU item will be removed when the cache
        # reached its maximum size.
        maxsize = kwargs.get('access_token_maxsize', 1000)
        # Given access token from Azure is good for 1 hour, keep in cache for 55 minutes by default
        ttl = kwargs.get('access_token_ttl', 3300)
        refresh_maxsize = kwargs.get('refresh_token_maxsize', 1000)
        # Given refresh token from Azure is good for 24 hours,
        # keep in cache for 23 hours 55 minutes by default
        refresh_ttl = kwargs.get('refresh_token_ttl', 79200)
        self._access_token_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._refresh_token_cache = TTLCache(maxsize=refresh_maxsize, ttl=refresh_ttl)

    @classmethod
    def get_instance(cls, **kwargs):
        return cls()


