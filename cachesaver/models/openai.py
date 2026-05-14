from openai import AsyncClient
from copy import deepcopy
from ..typedefs import Request, SingleRequestModel, Response
from .wrapper import AsyncWrapper, SyncWrapper, openai_chat_adapter


class OpenAIModel(SingleRequestModel):
    def __init__(self, **client_kwargs):

        self.client_kwargs = client_kwargs
        self.aclient = AsyncClient(**client_kwargs)

    async def request(self, request: Request):
        args = request.args
        kwargs = request.kwargs

        response = await self.aclient.chat.completions.create(
            *args, **kwargs, n=request.n
        )

        # Adjust token usage for n>1 by dividing completion tokens by n and recalculating total tokens
        token_usage = response.usage
        token_usage.completion_tokens //= request.n
        token_usage.total_tokens = token_usage.prompt_tokens + token_usage.completion_tokens

        # ToDo: this is difficult, how do we deal with the case n>1
        # Cachesaver then expects a list of replies
        flattened_responses = []
        for choice in response.choices:
            response_copy = deepcopy(response)
            response_copy.choices = [choice]
            response_copy.usage = token_usage
            flattened_responses.append(response_copy)

        return Response(
            data=flattened_responses)


class AsyncOpenAI(AsyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            batch_size=1,
            timeout=None,
            **client_kwargs):

        model = OpenAIModel(**client_kwargs)

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=batch_size,
            response_adapter=openai_chat_adapter,
            timeout=timeout
            )

class OpenAI(SyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            timeout=None,
            **client_kwargs
            ):

        model = OpenAIModel(**client_kwargs)

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=1,
            response_adapter=openai_chat_adapter,
            timeout=timeout
            )
