import transformers


def best_of_n(model, tokenizer, prompt, n=50, mbs=5):
    """Returns the best of n samples from a model
    :param model: A Huggingface model
    :param tokenizer: A Huggingface tokenizer
    :param prompt: A string prompt
    :param n: The number of samples to take
    :param mbs: The microbatch size
    :return: The best of n samples"""

    # first tokenize the prompt
    prompt = tokenizer(prompt, return_tensors="pt")
    print(prompt)
    # extract input_ids and attention_mask
    input_ids = prompt.input_ids
    attn_mask = prompt.attention_mask

    # and stack n times
    input_ids = input_ids.repeat(mbs, 1).to(model.device)
    attn_mask = attn_mask.repeat(mbs, 1).to(model.device)
    print(input_ids.shape)
    print(attn_mask.shape)

    # save a giant list of output_ids
    output_ids = []
    output_scores = []
    # now generate. make sure that we utilize our mbs
    for i in range(n // mbs):
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
        )

        # append to our list
        output_ids.append(out_temp.sequences)
        output_scores.append(out_temp.scores)

    zipped = zip(output_ids, output_scores)
    # sort by score in ascending order
    zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    # and return the best one
    return tokenizer.decode(zipped[0][0], skip_special_tokens=True)

if __name__ == "__main__":
    # load the model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    # now generate a response
    print(best_of_n(model, tokenizer, "Hello, my name is"))