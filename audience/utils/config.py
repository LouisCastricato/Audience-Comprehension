from dataclasses import dataclass
from pickle import LIST
from typing import Any, Dict, List

import yaml


@dataclass
class ExperimentConfig:
    # If this list is of length one, then we just use that one model for both agents.
    model_names : List[str] = ["GPTJAgent"]
    # The names of the agents in the experiment.
    agent_names : List[str] = ["Alice", "Bob"]
    # Determines the priors that we condition each agent on.
    prompt_pipeline : str = "DefaultPromptPipeline"
    # Determines the number of prompts that we condition each agent on. -1 means all.
    samples : int = -1
    # Determines the number of utterances that each agent generates.
    utterances : int = 10
    # Used for Best-Of-N sampling.
    best_of_n : int = 50
    # Determines the evaluation pipeline
    eval_pipeline : str = "DefaultEvalPipeline"
    # The device for each agent. If it is a list of length one, then we just use that one device for both agents.
    devices : List[str] = ["cuda:0"]

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)

@dataclass
class Config:
    experiment: ExperimentConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls(
            experiment.from_dict(config["experiment"]),
        )

    def to_dict(self):
        data = self.experiment.__dict__.copy()
        return data