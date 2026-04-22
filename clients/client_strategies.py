from abc import ABC, abstractmethod
import os
import dotenv

from cachesaver.models.openai import AsyncOpenAI as CacheSaverAsyncOpenAI
from cachesaver.models.groq import AsyncGroq as CacheSaverAsyncGroq
from cachesaver.typedefs import Metadata
from openai import AsyncOpenAI
from groq import AsyncGroq

def make_dummy_metadata(n=1):
    return Metadata(
        n=n,
        cached=False,
        duplicated=False
    )

def completion_with_metadata(client, messages, model, n):
    completion = client.chat.completions.create(
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
        self.client = CacheSaverAsyncOpenAI(
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
        self.client = AsyncOpenAI(
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
        self.client = CacheSaverAsyncOpenAI(
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
        self.client = AsyncOpenAI(
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
        self.client = CacheSaverAsyncGroq(
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
        self.client = AsyncGroq(
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