import sys
from typing import Dict, Iterable

from audience.data.utils import AgentBatch

# specifies a dictionary of models
_MODELS: Dict[str, any] = {}  # registry


def register_model(name):
    """Decorator used register models
    Args:
        name: Name of the models
    """

    def register_class(cls, name):
        _MODELS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


class BaseModel:
    def __init__(self):
        pass

    def generate_response(self, chat_transcript: Iterable[AgentBatch]):
        """
        Generate a response given a chat transcript and agent name
        :param chat_transcript: The chat thus far
        :param agent: Agent name that will be responding
        :return: The response
        """
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError


from audience.model.huggingface import GPTJAgent


def get_model(name):
    return _MODELS[name.lower()]


def get_model_names():
    return _MODELS.keys()
