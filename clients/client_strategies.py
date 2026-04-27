from abc import ABC, abstractmethod
import os
import dotenv

from cachesaver.models.openai import AsyncOpenAI as _CacheSaverAsyncOpenAI
from cachesaver.models.groq import AsyncGroq as _CacheSaverAsyncGroq
from cachesaver.typedefs import Metadata
from openai import AsyncOpenAI as _AsyncOpenAI
from groq import AsyncGroq as _AsyncGroq

def make_dummy_metadata(n=1):
    return Metadata(
        n=n,
        cached=[False for _ in range(n)],
        duplicated=[False for _ in range(n)]
    )

async def completion_with_metadata(client, messages, model, n):
    completion = await client.chat.completions.create(
                    messages=messages,
                    model=model,
                    n=n
    )
    metadata = make_dummy_metadata(n)
    return completion, metadata

def completion_with_metadata_cachesaver(client, messages, model, n):
    return client.chat.completions.create(
                    messages=messages,
                    model=model,
                    n=n,
                    metadata = True
    )


class ClientStrategy(ABC):
    @abstractmethod
    def create_chat_completion(self):
        pass

class CacheSaverOllamaClient(ClientStrategy):
    def __init__(self, model):
        self.client = _CacheSaverAsyncOpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama',  # required but ignored
            namespace="local_ollama_" + model,
            cachedir="./cache"
        )
        self.model = model

    def create_chat_completion(self, messages, n=1):
        return completion_with_metadata_cachesaver(
                    self.client,
                    messages=messages,
                    model=self.model,
                    n=n
        )

class OllamaClient(ClientStrategy):
    def __init__(self, model):
        self.client = _AsyncOpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama',  # required but ignored
        )
        self.model = model

    def create_chat_completion(self, messages, n=1):
        return completion_with_metadata(
                    self.client,
                    messages=messages,
                    model=self.model,
                    n=n
        )

class CacheSaverOpenAIClient(ClientStrategy):
    def __init__(self, model):
        self.client = _CacheSaverAsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            namespace="openai_" + model,
            cachedir="./cache"
        )
        self.model = model

    def create_chat_completion(self, messages, n=1):
        return completion_with_metadata_cachesaver(
                    self.client,
                    messages=messages,
                    model=self.model,
                    n=n
        )


class OpenAIClient(ClientStrategy):
    def __init__(self, model):
        self.client = _AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.model = model

    def create_chat_completion(self, messages, n=1):
        return completion_with_metadata(
                    self.client,
                    messages=messages,
                    model=self.model,
                    n=n
        )
    
class CacheSaverGroqClient(ClientStrategy):
    def __init__(self, model):
        dotenv.load_dotenv()
        self.client = _CacheSaverAsyncGroq(
            api_key = os.getenv("GROQ_API_KEY"),
            namespace="groq_" + model,
            cachedir="./cache",
        )
        self.model = model
    
    def create_chat_completion(self, messages, n=1):
        return completion_with_metadata_cachesaver(
                    self.client,
                    messages=messages,
                    model=self.model,
                    n=n
        )

class GroqClient(ClientStrategy):
    def __init__(self, model):
        dotenv.load_dotenv()
        self.client = _AsyncGroq(
            api_key = os.getenv("GROQ_API_KEY")
        )
        self.model = model
    
    def create_chat_completion(self, messages, n=1):
        return completion_with_metadata(
                    self.client,
                    messages=messages,
                    model=self.model,
                    n=n
        )