import asyncio
from anthropic import AsyncAnthropic as AsyncAnthropicClient
from ..typedefs import Request, SingleRequestModel, Response
from .wrapper import AsyncWrapper, SyncWrapper


class AnthropicModel(SingleRequestModel):
    def __init__(self, **client_kwargs):
        self.client_kwargs = client_kwargs
        self.aclient = AsyncAnthropicClient(**client_kwargs)

    async def request(self, request: Request) -> Response:
        args = request.args
        kwargs = request.kwargs

        assert request.n == 1, "Anthropic allows no 'n' parameter"

        # Claude doesn't support n>1 natively, so make n separate calls
        
        response = await self.aclient.messages.create(*args, **kwargs)
        print(f"Reponse:\n{response}")

        return Response(
            data=[response]
        )


class AsyncAnthropic(AsyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            batch_size=1,
            timeout=None,
            **client_kwargs):

        model = AnthropicModel(**client_kwargs)

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=batch_size,
            timeout=timeout
            )


class Anthropic(SyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            timeout=None,
            **client_kwargs
            ):

        model = AnthropicModel(**client_kwargs)

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=1,
            timeout=timeout
            )
