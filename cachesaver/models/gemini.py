import asyncio
from google import genai
from google.genai import types
from copy import deepcopy
from deepdiff import DeepHash
from dataclasses import asdict
from ..typedefs import Request, SingleRequestModel, Response
from .wrapper import AsyncWrapper, SyncWrapper, gemini_chat_adapter


class GeminiModel(SingleRequestModel):
    def __init__(self, **client_kwargs):
        self.client_kwargs = client_kwargs
        self.aclient = genai.Client(**client_kwargs)

    async def request(self, request: Request) -> Response:
        print(request)
        args = request.args
        kwargs = request.kwargs

        # Gemini doesn't reliably support candidate_count>1, make n separate calls
        kwargs_copy = kwargs.copy()
        if "config" in kwargs_copy:
            kwargs_copy["config"].candidate_count = request.n
        else:
            kwargs_copy["config"] = types.GenerateContentConfig(candidateCount=request.n)
        response = await self.aclient.aio.models.generate_content(*args, **kwargs_copy)

        flattened_responses = []
        for candidate in response.candidates:
            response_copy = deepcopy(response)
            response_copy.candidates = [candidate]
            flattened_responses.append(response_copy)

        return Response(
            data=flattened_responses
        )
    
    def hash(self, request: Request):
        params = asdict(request)
        del params["request_id"]
        del params["namespace"]
        del params["n"]
        if "config" in params["kwargs"]:
            params["kwargs"]["config"].candidate_count=None
        return DeepHash(params)[params]
        



class AsyncGemini(AsyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            batch_size=1,
            timeout=None,
            **client_kwargs):

        model = GeminiModel(**client_kwargs)

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=batch_size,
            response_adapter=gemini_chat_adapter,
            timeout=timeout
            )


class Gemini(SyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            timeout=None,
            **client_kwargs
            ):

        model = GeminiModel(**client_kwargs)

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=1,
            response_adapter=gemini_chat_adapter,
            timeout=timeout
            )
