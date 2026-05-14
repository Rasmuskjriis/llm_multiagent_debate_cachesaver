from openai import AsyncClient
from copy import deepcopy
from ..typedefs import Request, SingleRequestModel, Response
from .wrapper import AsyncWrapper, SyncWrapper, openai_chat_adapter


class VLLMModel(SingleRequestModel):
    def __init__(self, **client_kwargs):
        client_kwargs.setdefault("api_key", "EMPTY")
        self.client_kwargs = client_kwargs
        self.aclient = AsyncClient(**client_kwargs)

    async def request(self, request: Request) -> Response:
        args = request.args
        kwargs = request.kwargs

        response = await self.aclient.chat.completions.create(
            *args, **kwargs, n=request.n
        )

        flattened_responses = []
        for choice in response.choices:
            response_copy = deepcopy(response)
            response_copy.choices = [choice]
            flattened_responses.append(response_copy)

        return Response(
            data=flattened_responses)


class AsyncVLLM(AsyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            batch_size=1,
            timeout=None,
            **client_kwargs):

        model = VLLMModel(**client_kwargs)

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=batch_size,
            response_adapter=openai_chat_adapter,
            timeout=timeout
            )


class VLLM(SyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            timeout=None,
            **client_kwargs
            ):

        model = VLLMModel(**client_kwargs)

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=1,
            response_adapter=openai_chat_adapter,
            timeout=timeout
            )
