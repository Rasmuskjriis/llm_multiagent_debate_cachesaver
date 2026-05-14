from typing import NamedTuple, List, Protocol, Any, runtime_checkable, Optional
from dataclasses import dataclass, asdict
from deepdiff import DeepHash


class Metadata(NamedTuple):
    n: int
    cached: Optional[List[bool]] = None
    duplicated: Optional[List[bool]] = None


class PromptNamespace(NamedTuple):
    prompt: str
    namespace: str


@dataclass(frozen=True)
class Request:
    args: Any
    kwargs: Any
    n: int
    request_id: str
    namespace: str

    def __getattr__(self, name: str) -> None:
        return None

def hash(request: Request):
    """
    Defines the parameters that uniquely identify a request in the cache.

    A problem that can arise:
        - request made which specifies the temperature is 1.0 (default)
        - request made which doesn't specify the temperature (1.0 by default)
    These two requests will not be considered identical for caching purposes
    
    A similar problem can arise if they set temperature once through kwargs, and
    once through args.

    For both examples above, I just chose randmoly temperature, but this applies to
    any parameter that has a default value in the model API.

    Not sure how to fix this.
    """
    params = asdict(request)

     # Remove keys unrelated to the LLM's generation
    metadata = [
        "prompt_cache_key",
        "prompt_cache:retention",
        "safety_identifier",
        "service_tier",
        "store",
        "stream",
        "stream_options"
    ]

    # Remove metadata keys from kwargs before hashing
    for key in metadata:
        params["kwargs"].pop(key, None)
    del params["request_id"]
    del params["namespace"]
    del params["n"]
    return DeepHash(params)[params]


class Batch(NamedTuple):
    requests: List[Request]


class Response(NamedTuple):
    data: List[Any]
    cached: Optional[List[bool]] = None
    duplicated: Optional[List[bool]] = None


@runtime_checkable
class SingleRequestModel(Protocol):
    """Model that processes one request at a time, can return multiple results."""

    async def request(self, request: Request) -> Response: ...


@runtime_checkable
class BatchRequestModel(Protocol):
    """Model that processes multiple requests at once, returns multiple results per request."""

    async def batch_request(self, batch: Batch) -> List[Response]: ...
