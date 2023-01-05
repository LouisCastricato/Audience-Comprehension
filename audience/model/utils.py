import torch
import transformers
from tqdm import tqdm

bad_word_list = ["\n\n"]


def permute_bad_words(bad_word_list):
    """
    For every word, adds a space in front and appends back
    """
    new_bad_word_list = []
    for bad_word in bad_word_list:
        new_bad_word_list.append(" " + bad_word)
    return new_bad_word_list + bad_word_list


def best_of_n(
    model,
    tokenizer,
    prompt,
    n=100,
    top_k=1,
    mbs=20,
    max_length=100,
    use_tqdm=False,
    bad_words=None,
):
    """Returns the best of n samples from a model
    :param model: A Huggingface model
    :param tokenizer: A Huggingface tokenizer
    :param prompt: A string prompt
    :param n: The number of samples to take
    :param top_k: The number of top k samples to take
    :param mbs: The microbatch size
    :param max_length: The maximum length of the output
    :param use_tqdm: Whether to use tqdm
    :param bad_words: A list of bad words to filter out
    :return: The best of n samples"""

    # first tokenize the prompt
    prompt = tokenizer(prompt, return_tensors="pt")

    # extract input_ids and attention_mask
    input_ids = prompt.input_ids
    attn_mask = prompt.attention_mask

    # accomodate for prompt length
    adjusted_length = input_ids.shape[1] + max_length

    # and stack n times
    input_ids = input_ids.repeat(mbs, 1).to(model.device)
    attn_mask = attn_mask.repeat(mbs, 1).to(model.device)

    # save a giant list of output_ids
    output_ids = []
    output_scores = []

    # iterate over the number of samples
    if use_tqdm:
        iterator = tqdm(range(n // mbs))
    else:
        iterator = range(n // mbs)

    # now generate. make sure that we utilize our mbs
    for i in iterator:
        # generate
        out_temp = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_length=adjusted_length,
            do_sample=True,
            top_p=0.95,
            top_k=60,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("\n")[0],
            bad_words_ids=bad_words,
        )
        # get the length of input_ids
        start_idx = input_ids.shape[1]

        # compute perplexity
        log_probs = []
        lengths = [0] * mbs
        for i, t in enumerate(out_temp.scores):
            # log softmax
            t = -torch.log_softmax(t, dim=-1)[:, out_temp.sequences[:, start_idx + i]][
                0
            ]

            # compute lengths
            for j, t_j in enumerate(t):
                if not (t_j == float("inf")):
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
    outputs = []
    for output_ids in output_ids[:top_k]:
        outputs.append(
            tokenizer.decode(torch.tensor(output_ids), skip_special_tokens=True)
        )
    return outputs

def remove_eol_add_suffix(input_string : str, suffix : str) -> str:
    """
    Removes a trailing EOL, if applicable, and appends the suffix
    :param input_string: The input string
    :param suffix: The suffix to add
    :return: The output string
    """
    # removes double newlines
    str_arr = input_string.split("\n")
    # remove empty strings
    str_arr = [x for x in str_arr if x != ""]
    # join with \n
    output_string = "\n".join(str_arr)
    # append suffix
    input_string += suffix
    return input_string