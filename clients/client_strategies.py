from abs import ABC, abstractmethod

from cachesaver.models.openai import AsyncOpenAI

class ClientStrategy(ABC):
    @abstractmethod
    def create_chat_completion(self):
        pass

class LocalOllamaCLient(ClientStrategy):
    def __init__(self, model, cachesaver=True):
        self.client = AsyncOpenAI(
            base_url='http://localhost:11434/v1/',
            api_key='ollama',  # required but ignored
            namespace="local_ollama_" + model,
            cachedir="../cache" if cachesaver else None
        )
        self.model = model

    def create_chat_completion(self, messages, n=1):
        return self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    n=n)