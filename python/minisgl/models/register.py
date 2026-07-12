import importlib

from .config import ModelConfig

_MODEL_REGISTRY = {
    "DeepseekV4ForCausalLM": (".deepseek_v4", "DeepseekV4ForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model architecture {model_architecture!r} is not supported; "
            "this release supports DeepSeek V4 Flash only "
            "(DeepseekV4ForCausalLM)."
        )
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
