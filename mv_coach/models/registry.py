"""Model registry for versioned model management."""

from typing import Callable, Dict

import torch.nn as nn

from mv_coach.core.exceptions import ModelError
from mv_coach.models.backbone import CNN1DBackbone, TCNBackbone


class ModelRegistry:
    """Registry for model architectures."""

    _registry: Dict[str, Callable[..., nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a model.

        Args:
            name: Model name.

        Returns:
            Decorator function.
        """

        def decorator(model_class: Callable[..., nn.Module]) -> Callable:
            cls._registry[name] = model_class
            return model_class

        return decorator

    @classmethod
    def build(cls, name: str, **kwargs) -> nn.Module:
        """Build a model by name.

        Args:
            name: Model name.
            **kwargs: Model parameters.

        Returns:
            Model instance.

        Raises:
            ModelError: If model not found.
        """
        if name not in cls._registry:
            raise ModelError(
                f"Model '{name}' not found in registry. "
                f"Available models: {list(cls._registry.keys())}"
            )

        return cls._registry[name](**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered models.

        Returns:
            List of model names.
        """
        return list(cls._registry.keys())


# Register built-in models
ModelRegistry.register("tcn")(TCNBackbone)
ModelRegistry.register("cnn1d")(CNN1DBackbone)
