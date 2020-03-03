from abc import ABCMeta, abstractmethod
from typing import List, Dict, Any, Collection, Callable


class TextIndexTranslator:
    def __init__(self, name: str, eos_id: int):
        self.name = name
        self.eos_id = eos_id

    @abstractmethod
    def encode(self, text) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass
