from dataclasses import dataclass, field
from transformers.tokenization_utils_base import BatchEncoding
from typing import List, Tuple, Callable, Any
from torchtyping import TensorType
from typeguard import typechecked

# agent batch defines the input to an agent. It gives us a prior, the agent name, and a list of the dialogue
# so far, per agent.
@dataclass
class AgentBatch:
    # The prior for this agent.
    prior: List[str]
    # The name of the agent.
    agent_name: str
    # The precondition dialogue
    precondition: str
    # The dialogue so far, per agent. (Agent name, utterance)
    dialogue: List[Tuple[str, str]] = field(default_factory=list)

    # The prompt to be fed to the language model. This is computed below. Do not manually set this.
    prompt: str = None

    # Tokenized data
    input_ids: TensorType["batch_size", "seq_len"] = None
    attention_mask: TensorType["batch_size", "seq_len"] = None




def construct_prompt(inp : AgentBatch) -> str:
    """
    Constructs a prompt for the given agent batch.
    :param inp: The agent batch.
    :return: The prompt.
    """
    # start by setting the prompt to the prior
    prompt = inp.prior + "\n\n"
    # add the precondition
    prompt += inp.precondition

    # check if we've accumulated any dialogue thus far
    if len(inp.dialogue) > 0:
        # then add the dialogue so far
        for agent_name, utterances in inp.dialogue:
            prompt += f"{agent_name}: {' '.join(utterances)}\n"
    # then add the agent name
    prompt += f"{inp.agent_name}: "
    
    return prompt


def _construct_prompt(inp : AgentBatch, _tok : Callable = None) -> AgentBatch:
    """
    Constructs a prompt for the given agent batch.
    :param inp: The agent batch.
    :param _tok: The tokenizer.
    :return: The prompt.
    """
    if _tok is None:
        inp.prompt = construct_prompt(inp)    
    else:
        # if we are going to tokenize, make sure input ids and attention mask is copied over
        prompt = construct_prompt(inp)
        input_dict = _tok(prompt)

        inp.prompt = prompt
        inp.input_ids = input_dict["input_ids"]
        inp.attention_mask = input_dict["attention_mask"]

    return inp

def create_tok(tokenizer: Callable, context_len: int):
    @typechecked
    def _tok(string_batch: Iterable[str]) -> BatchEncoding:
        if not isinstance(string_batch, list):
            string_batch = list(string_batch)
        return tokenizer(string_batch, padding="max_length", truncation=True, max_length=context_len)

    return _tok
