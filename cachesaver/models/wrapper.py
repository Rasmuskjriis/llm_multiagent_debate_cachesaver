import asyncio
import nest_asyncio
from ..pipelines import OnlineAPI, LocalAPI
from diskcache import Cache
from copy import deepcopy
from ..typedefs import Request, Metadata, SingleRequestModel, hash


def openai_chat_adapter(flattened_responses):
    """Adapter for OpenAI chat completion responses.

    Merges choices from flattened individual responses back into a single
    response object with all choices combined.

    Merges token usage across all responses, adjusting completion tokens 
    and total tokens accordingly.
    """
    choices = [resp.choices[0] for resp in flattened_responses.data]
    one_response = deepcopy(flattened_responses.data[0])
    one_response.choices = choices

    usage = flattened_responses.data[0].usage
    usage.completion_tokens = sum([resp.usage.completion_tokens for resp in flattened_responses.data])
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
    one_response.usage = usage

    return one_response

def gemini_chat_adapter(flattened_responses):
    """Adapter for Gemini chat completion responses.

    Merges candidates from flattened individual responses back into a single
    response object with all candidates combined.
    """
    candidates = [resp.candidates[0] for resp in flattened_responses.data]
    one_response = deepcopy(flattened_responses.data[0])
    one_response.candidates = candidates
    return one_response


def default_adapter(flattened_responses):
    """Generic adapter for providers that don't use the OpenAI choices format.

    Returns the single response for n=1, or a list of responses for n>1.
    """
    data = flattened_responses.data
    if len(data) == 1:
        return data[0]
    return data


class Create:
    def __init__(self, pipeline, namespace, response_adapter=None):
        self.pipeline = pipeline
        self.global_request_counter = 0
        self.namespace = namespace
        self.response_adapter = response_adapter or default_adapter

    async def async_call(
            self,
            *args,
            namespace: str = None,
            request_id: str = None,
            metadata: bool = False,
            **kwargs):
        
        self.global_request_counter += 1 # Async safe?

        # Number of output samples requested
        if "n" in kwargs:
            # OpenAI API
            n = kwargs.pop("n")
        elif "config" in kwargs:
            # Gemini API 
            n=kwargs["config"].candidate_count
            kwargs["config"].candidate_count=1
        elif "num_return_sequences" in kwargs:
            # Transformers API
            n=kwargs.pop("num_return_sequences")
        else:
            n = 1

        request = Request(
                args=args,
                kwargs=kwargs,
                n=n,
                namespace=namespace or self.namespace,
                request_id=request_id or str(self.global_request_counter)
            )
        
        # Responses are flattened due to CS pipeline
        flattened_responses = await self.pipeline.request(
            request= request
        )

        # Adapt the flattened responses back into the format expected by the client
        adapted_responses = self.response_adapter(flattened_responses)
        
        if metadata:
            # Generate metadata in regards to caching and duplication
            metadata = Metadata(
                n= len(flattened_responses.data),
                cached = flattened_responses.cached,
                duplicated = flattened_responses.duplicated
            )
            return adapted_responses, metadata
        else:
            return adapted_responses
            
        

class AsyncCreate(Create):
    def __init__(self, api, namespace, response_adapter=None):
        super().__init__(api, namespace, response_adapter)

    async def __call__(
            self, 
            *args, 
            namespace: str = None, 
            request_id: str = None,
            metadata: bool = False,
            **kwargs
            ):
        return await self.async_call(
            *args, 
            namespace=namespace, 
            request_id=request_id,
            metadata=metadata, 
            **kwargs
            )

class SyncCreate(Create):
    def __init__(self, api, namespace, response_adapter=None):
        super().__init__(api, namespace, response_adapter)

    def __call__(
            self, 
            *args, 
            namespace: str = None, 
            request_id: str = None,
            metadata: bool = False,
            **kwargs):
        nest_asyncio.apply()
        result = asyncio.run(
            self.async_call(
                *args, 
                namespace=namespace, 
                request_id=request_id,
                metadata=metadata,
                **kwargs))
        return result


class APIProxy:
    """Generic attribute proxy that routes any attribute chain to the same create function.

    This allows client.chat.completions.create(...), client.messages.create(...),
    client.responses.create(...), etc. to all work through the same underlying call.
    """
    def __init__(self, create_fn: Create):
        self._create_fn = create_fn

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return APIProxy(self._create_fn)

    def __call__(self, *args, **kwargs):
        return self._create_fn(*args, **kwargs)


class Wrapper:
    def __init__(self,
                 model: SingleRequestModel,
                 namespace="default",
                 cachedir="./cache",
                 batch_size=1,
                 response_adapter=default_adapter,
                 pipeline="online",
                 timeout=None
                ):
        self.namespace = namespace
        self.batch_size = batch_size
        self.cachedir = cachedir
        self.cache = Cache(
            self.cachedir,
            timeout=-1
        )
        self.model = model
        self.response_adapter = response_adapter

        try:
            self.hash = model.hash
        except AttributeError:
            self.hash = hash

        if pipeline == "online":
            self.pipeline = OnlineAPI(
                model=self.model,
                cache=self.cache,
                batch_size=self.batch_size,
                hash=self.hash,
                timeout=timeout or -1,
                allow_batch_overflow=True
            )
        elif pipeline == "local":
            self.pipeline = LocalAPI(
                model=self.model,
                cache=self.cache,
                batch_size=self.batch_size,
                hash=self.hash,
                timeout=timeout or -1
            )
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline!r}. Use 'online' or 'local'.")

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return APIProxy(self._create)


class AsyncWrapper(Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create = AsyncCreate(self.pipeline, self.namespace, self.response_adapter)
        self.chat = APIProxy(self._create)


class SyncWrapper(Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create = SyncCreate(self.pipeline, self.namespace, self.response_adapter)
        self.chat = APIProxy(self._create)
