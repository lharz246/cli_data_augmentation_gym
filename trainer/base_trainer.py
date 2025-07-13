from abc import ABC, abstractmethod
from typing import Dict, Any, Type


class BaseTrainer(ABC):
    _registry: Dict[str, Type["BaseTrainer"]] = {}

    def __init__(self, config: Any):
        self.config = config

    @classmethod
    def register(cls, name: str):
        def decorator(trainer_class: Type["BaseTrainer"]):
            cls._registry[name] = trainer_class
            return trainer_class

        return decorator

    @classmethod
    def create(cls, trainer_type: str, config: Any) -> "BaseTrainer":
        if trainer_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown trainer type: {trainer_type}. Available: {available}"
            )
        trainer_class = cls._registry[trainer_type]
        return trainer_class(config)

    @classmethod
    def list_available(cls) -> list:
        return list(cls._registry.keys())

    @abstractmethod
    def prepare_epoch(self, epoch: int):
        pass

    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def validation_step(self, batch):
        pass

    @abstractmethod
    def on_training_start(self):
        pass

    @abstractmethod
    def on_training_end(self):
        pass
