from abc import ABCMeta, abstractmethod
from typing import List, Dict, Any, Collection, Callable


class TextPreprocessor:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def pre_process_doc(self, doc: str) -> str:
        pass
