import asyncio
import functools
from ..typedefs import Request, Batch, Response, SingleRequestModel
from .wrapper import SyncWrapper
from deepdiff import DeepHash
from dataclasses import asdict

try:
    import torch
    from transformers import AutoModelForCausalLM as AutoModelClient
except ImportError:
    torch = None
    AutoModelClient = None


class HFTransformersModel(SingleRequestModel):
    def __init__(self, *model_args, **model_kwargs):
        if AutoModelClient is None:
            raise ImportError(
                "HF Transformers requires 'transformers' and 'torch'. "
                "Install them with: pip install cachesaver[transformers]"
            )

        self.model = AutoModelClient.from_pretrained(
            *model_args,
            **model_kwargs
        )

    async def request(self, request: Request):
        args= request.args
        kwargs = request.kwargs

        with torch.no_grad():
            response = self.model.generate(
                *args,
                **kwargs,
                num_return_sequences=request.n
            )
        
        flattened_responses = [r for r in response]

        return Response(
            data=flattened_responses
        )
    
    def hash(self, request: Request):
        params = asdict(request)
        del params["request_id"]
        del params["namespace"]
        del params["n"]
        params = make_hashable(params)
        return DeepHash(params)[params]


class AutoModelForCausalLM(SyncWrapper):
    def __init__(
            self,
            *model_args,
            namespace="default",
            cachedir="./cache",
            batch_size=8,
            timeout=None,
            **model_kwargs
            ):

        model = HFTransformersModel(
            *model_args,
            **model_kwargs
        )

        super().__init__(
            model=model,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=batch_size,
            response_adapter=transformers_chat_adapter,
            pipeline="online",
            timeout=timeout
        )
    
    @classmethod
    def from_pretrained(
            cls,
            *model_args,
            namespace="default",
            cachedir="./cache",
            batch_size=8,
            timeout=None,
            **model_kwargs
            ):
        return cls(
            *model_args,
            namespace=namespace,
            cachedir=cachedir,
            batch_size=batch_size,
            timeout=timeout,
            **model_kwargs
        )

def transformers_chat_adapter(flattened_responses):
    """Adapter for Hugging Face transformers chat completion responses.

    Merges candidates from flattened individual torch tensors back into a single stacked tensor.
    """
    one_response = torch.stack(flattened_responses.data, dim=0)
    return one_response

def make_hashable(obj):
    if isinstance(obj, torch.Tensor):
        return {
            "__tensor__": True,
            "dtype": str(obj.dtype),
            "shape": tuple(obj.shape),
            "data": obj.cpu().numpy().tolist(),
        }

    elif isinstance(obj, dict):
        return {k: make_hashable(v) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        return [make_hashable(v) for v in obj]

    return obj