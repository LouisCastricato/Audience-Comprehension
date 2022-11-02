from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


@dataclass
class ExperimentConfig:
    # If this list is of length one,
    # then we just use that one model for both agents.
    model_names: List[str] = field(default_factory=list)
    # The names of the agents in the experiment.
    agent_names: List[str] = field(default_factory=list)
    # Determine our collation function. This allows
    # for intervention and prompting.
    data_pipeline: str = "BaseDataPipeline"
    # Determines the number of prompts that we condition
    # each agent on. -1 means all.
    samples: int = -1
    # Determines the number of utterances that each agent generates.
    utterances: int = 10
    # Used for Best-Of-N sampling.
    best_of_n: int = 50
    # Determines the evaluation pipeline
    eval_pipeline: str = "BaseEvalPipeline"
    # The device for each agent. If it is a list
    # of length one, then we just use that one device for both agents.
    devices = "cuda:0"
    # directory of the dataset. Only used for the BaseDataPipeline
    dataset_dir: str = None

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
            ExperimentConfig.from_dict(config["ExperimentConfig"]),
        )

    def to_dict(self):
        data = self.experiment.__dict__.copy()
        return data
