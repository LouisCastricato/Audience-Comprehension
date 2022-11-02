from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from audience.data.utils import AgentBatch
from audience.model import BaseModel, register_model


@register_model
class GPTJAgent(BaseModel):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    def generate_response(self, chat_transcript: List[AgentBatch]):
        """
        Generates a response for a batch of conversations.
        :param chat_transcript: A list of AgentBatch objects, each of which contains a conversation.
        :returns: A list of strings, each of which is a response to the conversation
        """
        # fetch the input ids and attention mask from the agent batch
        input_ids, attn_mask = chat_transcript.input_ids, chat_transcript.attention_mask
        output_ids = self.model.generate(input_ids, max_length=100, do_sample=True)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def get_model(self):
        return self.model


@register_model
class GPT2Agent(BaseModel):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def generate_response(self, chat_transcript: List[AgentBatch]):
        """
        Generates a response for a batch of conversations.
        :param chat_transcript: A list of AgentBatch objects, each of which contains a conversation.
        :returns: A list of strings, each of which is a response to the conversation
        """
        # fetch the input ids and attention mask from the agent batch
        input_ids, attn_mask = chat_transcript.input_ids, chat_transcript.attention_mask
        output_ids = self.model.generate(input_ids, max_length=100, do_sample=True)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def get_model(self):
        return self.model
