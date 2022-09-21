from audience.model import register_model, BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

@register_model
class GPTJAgent(BaseModel):
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    def generate_response(self, chat_transcript: str, agent: str):
        #input_ids = self.tokenizer(input_prompt + "\n" + , return_tensors="pt").input_ids
        output_ids = self.model.generate(input_ids, max_length=100, do_sample=True)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def get_model(self):
        return self.model