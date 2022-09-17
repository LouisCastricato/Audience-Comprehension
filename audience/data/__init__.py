from typing import List, Tuple, Callable, Any
from utils import AgentBatch, construct_prompt, _construct_prompt, create_tok
from functools import partial

# specifies a dictionary of models
_DATAPIPELINES: Dict[str, any] = {}  # registry


def register_datapipeline(name):
    """Decorator used register models
    Args:
        name: Name of the models
    """

    def register_class(cls, name):
        _DATAPIPELINES[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


@register_datapipeline
class BaseDataPipeline(Dataset):
    """Dataset wrapper class to ease working with the audience dataset and Pytorch data utilities."""

    def __init__(
        self,
        path: str = "dataset.csv",
    ):
        # data.csv is in the format: prior 1, prompt 1, prior 2 prompt 2
        self.data = pd.read_csv(path, header=None)


    def __getitem__(self, index: int) -> Tuple[AgentBatch, AgentBatch]:
        # get the alice and bob agent batches
        alice_batch = AgentBatch(
            prior=self.data.iloc[index, 0],
            agent_name="alice",
            precondition=self.data.iloc[index, 1],
        )
        bob_batch = AgentBatch(
            prior=self.data.iloc[index, 2],
            agent_name="bob",
            precondition=self.data.iloc[index, 3],
        )

        return alice_batch, bob_batch
    

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def create_factories(
        call_tokenizer: Callable, tokenizer_factory: Callable, context_len: int = 2048
    ) -> Tuple[Callable, Callable]:

        """Function creates a callable tokenizer subroutine and uses it to curry the tokenizer factory
        Args:
            call_tokenizer (Callable): A function defined within BaseEncoder that outlines a custom encoder processing step
            tokenizer_factory (Callable): The factory we wish to initialize
            context_len (int): Max context length of a batch element.
        Returns:
            Callable: A tuple of functions that perform initial tokenization and update tokenization
        """
        tok_func = create_tok(call_tokenizer, context_len=context_len)
        return partial(precondition_factory, tok_func), partial(update_factory, tok_func)

    @staticmethod
    def precondition_factory(_tok: Callable) -> Callable:

        """Function factory that creates a collate function for use with a torch.util.data.Dataloader
        Args:
            _tok (Callable): A Huggingface model tokenizer, taking strings to torch Tensors
        Returns:
            Callable: A function that will take a batch of string tuples and tokenize them properly.
        """

        @typechecked
        def collate(
            data: Iterable[Tuple[AgentBatch, AgentBatch]]
        ) -> Iterable[Tuple[AgentBatch, AgentBatch]]:
            # store the agent batches
            agent_batches = []

            # iterate over the conversations
            for conversation in data:
                alice_batch, bob_batch = conversation

                # tokenize
                alice_batch = _construct_prompt(alice_batch, _tok)
                bob_batch = _construct_prompt(bob_batch, _tok)

                # append
                agent_batches.append((alice_batch, bob_batch))
            return agent_batches

        return collate


    @staticmethod
    def update_factory(_tok: Callable) -> Callable:
        
        """Similar to the above, this collates incoming dialogue update
        Args:
            _tok (Callable): A Huggingface model tokenizer, taking strings to torch Tensors
        Returns:
            Callable: A function that will take an agent batch and update them properly.
        """

        @typechecked
        def collate(batch : AgentBatch, agent : str, response : str) -> AgentBatch:
            """Update the dialogue with the response
            Args:
                batch (AgentBatch): The current dialogue
                agent (str): The agent who is responding
                response (str): The response
            Returns:
                AgentBatch: The updated dialogue
            """
            # update the dialogue
            batch.dialogue.append((agent, response))

            # update the prompt
            batch = _construct_prompt(batch, _tok)

            # return the updated dialogue
            return batch
        
        return collate
