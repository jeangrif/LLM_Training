from abc import ABC, abstractmethod

class BaseProvider(ABC):
    """Common interface for all model provider"""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """generate response from a prompt"""
        pass