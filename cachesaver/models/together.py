from together import AsyncClient
from copy import deepcopy
from ..typedefs import Request, SingleRequestModel, Response
from .wrapper import AsyncWrapper, SyncWrapper, openai_chat_adapter


class TogetherAIModel(SingleRequestModel):
    def __init__(self):
        self.aclient = AsyncClient()

    async def request(self, request: Request):
        args = request.args
        kwargs = request.kwargs

        response = await self.aclient.chat.completions.create(
            *args, **kwargs, n=request.n
        )

        # ToDo: this is difficult, how do we deal with the case n>1
        # Cachesaver then expects a list of replies
        flattened_responses = []
        for choice in response.choices:
            response_copy = deepcopy(response)
            response_copy.choices = [choice]
            flattened_responses.append(response_copy)

        return Response(
            data=flattened_responses)


class AsyncTogether(AsyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            batch_size=1,
            timeout=None):

        model = TogetherAIModel()

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=batch_size,
            response_adapter=openai_chat_adapter,
            timeout=timeout)


class Together(SyncWrapper):
    def __init__(
            self,
            namespace="default",
            cachedir="./cache",
            batch_size=1,
            timeout=None):

        model = TogetherAIModel()

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=batch_size,
            response_adapter=openai_chat_adapter,
            timeout=timeout
        )