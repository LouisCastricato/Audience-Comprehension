from audience.model.utils import *

if __name__ == "__main__":
    # load the model and tokenizer. GPT-J
    model = (
        transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").half().to("cuda")
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    prompts = [
        "Below is a conversation between two people. Bob and Alice make small talk.\nA: Hi Bob!\nB: How are you doing?\nA:",
        "Below is a conversation between Alice and Bob. Alice and Bob are discussing a new movie.\nA: Have you seen the new movie?\nB: No, I haven't. What's it about?\n\A:",
        "Below is a conversation between Alice and Bob. Alice and Bob are discusing the weather.\nA: It's been raining a lot lately.\nB: Yeah, I know. I hate the rain. Where are you going?\nA:",
        "Below is a conversation between Alice and Bob. Alice and Bob are discussing a new restaurant.\nA: Have you been to the new restaurant?\nB: No, I haven't. What's it like?\nA:",
        "Alice and Bob are making small talk.\nA: Hi Bob!\nB: How are you doing?\nA:",
    ]

    # tokenize bad word list
    bad_words_tokenized = tokenizer(permute_bad_words(bad_word_list)).input_ids
    # now generate a response

    # generate 5 utterances
    current_agent_idx=0
    for idx in tqdm(range(5)):
        outputs = []
        for prompt in tqdm(prompts):
            generations = best_of_n(
                model, tokenizer, prompt, max_length=25, top_k=2, bad_words=bad_words_tokenized, mbs=20, n=200
            )
            # Append the correct agent name
            if current_agent_idx%2==0:
                outputs = [remove_eol_add_suffix(x, "B:") for x in generations] + outputs
            else:
                outputs = [remove_eol_add_suffix(x, "A:") for x in generations] + outputs
            current_agent_idx+=1
        prompts = outputs
        # save to a csv, \n\n\n delimited
        with open("outputs_" + str(idx) + ".csv", "w") as f:
            f.write("\n\n\n".join(outputs))