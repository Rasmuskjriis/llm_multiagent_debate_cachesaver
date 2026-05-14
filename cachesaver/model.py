from openai import AsyncClient
from .pipelines import OnlineAPI
from diskcache import Cache
from copy import deepcopy
from .typedefs import Request, SingleRequestModel, Response


class OpenAIModel(SingleRequestModel):
    def __init__(self, aclient: AsyncClient):
        self.aclient = aclient

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


class Create:
    def __init__(self, aclient, api, namespace):
        self.aclient = aclient
        self.api = api
        self.global_request_counter = 0
        self.namespace = namespace

    async def __call__(self, *args, **kwargs):
        self.global_request_counter += 1
        if "n" in kwargs:
            n = kwargs.pop("n")
        else:
            n = 1
        flattened_responses = await self.api.request(Request(
            args=args,
            kwargs=kwargs,
            n=n,
            namespace=self.namespace,
            request_id=str(self.global_request_counter)
        ))
        choices = [resp.choices[0] for resp in flattened_responses.data]
        one_response = deepcopy(flattened_responses.data[0])
        one_response.choices = choices
        return one_response


class Completions:
    def __init__(self, aclient, api, namespace):
        self.create = Create(aclient, api, namespace)


class Chat:
    def __init__(self, aclient, api, namespace):
        self.completions = Completions(aclient, api, namespace)


class AsyncWrapper:
    def __init__(self,
                 namespace="default",
                 cachedir="./cache",
                 batch_size=1):
        self.namespace = namespace
        self.cachedir = cachedir
        self.batch_size = batch_size
        self.cache = Cache("./cache", timeout=-1)
        self.aclient = AsyncClient()
        self.model = OpenAIModel(self.aclient)
        self.api = OnlineAPI(model=self.model,
                             cache=self.cache, batch_size=self.batch_size)
        self.chat = Chat(self.aclient, self.api, self.namespace)
