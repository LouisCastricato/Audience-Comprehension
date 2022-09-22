import torch
import transformers
from tqdm import tqdm

def best_of_n(model, tokenizer, prompt, n=100, mbs=20, use_tqdm=False):
    """Returns the best of n samples from a model
    :param model: A Huggingface model
    :param tokenizer: A Huggingface tokenizer
    :param prompt: A string prompt
    :param n: The number of samples to take
    :param mbs: The microbatch size
    :return: The best of n samples"""

    # first tokenize the prompt
    prompt = tokenizer(prompt, return_tensors="pt")

    # extract input_ids and attention_mask
    input_ids = prompt.input_ids
    attn_mask = prompt.attention_mask

    # and stack n times
    input_ids = input_ids.repeat(mbs, 1).to(model.device)
    attn_mask = attn_mask.repeat(mbs, 1).to(model.device)


    # save a giant list of output_ids
    output_ids = []
    output_scores = []

    # iterate over the number of samples
    if use_tqdm:
        iterator = tqdm(range(n//mbs))
    else:
        iterator = range(n//mbs)
        
    # now generate. make sure that we utilize our mbs
    for i in iterator:
        # generate
        out_temp = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_length=100,
            do_sample=True,
            top_p=0.95,
            top_k=60,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=50256,
        )
        # get the length of input_ids
        start_idx = input_ids.shape[1]

        # compute perplexity
        log_probs = []
        lengths = [0] * mbs
        for i, t in enumerate(out_temp.scores):
            # log softmax
            t = -torch.log_softmax(t, dim=-1)[:, out_temp.sequences[:, start_idx+i]][0]

            # compute lengths
            for j, t_j in enumerate(t):
                if not(t_j == float("inf")):
                    lengths[j] += 1

            # replace inf with 0
            t[t == float("inf")] = 0

            log_probs.append(t)

        # save input_ids
        output_ids += out_temp.sequences.tolist()

        # stack log probs
        log_probs = torch.stack(log_probs, dim=1).sum(dim=1)
        # divide by length
        log_probs = log_probs / torch.tensor(lengths).to(log_probs.device)

        output_scores += log_probs.tolist()

    # zip for sorting
    zipped = zip(output_ids, output_scores)
    # sort by score in ascending order
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)

    # unzip
    output_ids, output_scores = zip(*zipped)

    # and return the best one
    return tokenizer.decode(torch.tensor(output_ids[0]), skip_special_tokens=True)

if __name__ == "__main__":
    # load the model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B').to("cuda").half()
    tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    prompt = """Below is a conversation between two people, Alice and Bob. Alice and Bob want to make small talk. Bob accuses Alice of sleeping with his wife.
A: Hi Bob!
B: How are you doing?
"""
    # now generate a response
    print(best_of_n(model, tokenizer, prompt, use_tqdm=True))
