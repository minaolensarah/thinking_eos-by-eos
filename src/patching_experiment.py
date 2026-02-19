
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM 
import torch 
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
import csv
import argparse



# sampling for LLaDA1.5
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

# sampling for Dream-v0
def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    """
    From logits to tokens for dream
    """

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0
    
def main(task, modelname):

    if "2.0" in modelname:
        model = AutoModelForCausalLM.from_pretrained(modelname, trust_remote_code=True, device_map="auto")
    else:
        model = AutoModel.from_pretrained(modelname, trust_remote_code=True, device_map="auto")

    model.tie_weights()

    tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=True, trust_remote_code=True)

    if "1.5" in modelname:
        tokenizer.mask_token_id = 126336
        end_of_turn_token = "<|eot_id|>"
    elif "dream" in modelname.lower():
        end_of_turn_token = "<|im_end|>"
    elif "2.0" in modelname:
        end_of_turn_token = "<|role_end|>"

    print("mask token id:", tokenizer.mask_token_id)
    print("eos token id:", tokenizer.eos_token_id)
    numlayers = model.config.num_hidden_layers
    print("number of layers:", numlayers)
    # load dataset of clean - corrputed data pairs

    if args.task == "maths":
        pairs1 = pd.read_json("./datasets/easy_maths_200perCalc.jsonl", orient='records', lines=True)
        pairs2 = pairs1.copy()
        def corrupt(calculation):
            q = ""
            for l in calculation:
                if l == "-":
                    l = "+"
                elif l == "+":
                    l = "-"
                q += l
            return q
        pairs2["calculation"] = pairs2["calculation"].apply(corrupt)
        sysprompt = {"role": "system", "content":"Answer the question with 'The final result is ...'. Do not give any additional explanation."}
        for df in [pairs1, pairs2]:
            df["sentence_id"] = df.index
            df["prompt"] = df.apply(lambda x:  [sysprompt, {"role": "user", "content": f"What is the result of {x['calculation']}?"},
            {"role": "assistant", "content": "The final result is " }], axis=1)
            df["prompt"] = df["prompt"].apply(lambda m: tokenizer.apply_chat_template(m, tokenize=False).rpartition(end_of_turn_token)[0])
        print("Example prompt: ", pairs1["prompt"].iloc[0])
        print("Example prompt: ", pairs2["prompt"].iloc[0])
        NUM_MASKS = 4

    if args.task == "boxes":
        pairs1 = pd.read_json("./datasets/boxes_testset_24X30_pair1.jsonl", orient='records', lines=True)
        pairs2 = pd.read_json("./datasets/boxes_testset_24X30_pair2.jsonl", orient='records', lines=True)

        for pairs in [pairs1, pairs2]:
            pairs["sentence_masked"] = pairs["sentence_masked"].apply(lambda s: s.replace("<extra_id_0> .", ""))
            sysprompt = {"role": "system", "content":"Answer the question but do not give any additional explanation."}
            pairs["prompt"] = pairs["sentence_masked"].apply(lambda sample: [sysprompt, {"role": "user", "content": sample.rpartition(".")[0] + "." + "\nWhat does Box " + sample.rpartition(".")[2].split("Box ")[1].split(" ")[0] + " contain?" },
                    {"role": "assistant", "content": "Box " + sample.rpartition(".")[2].split("Box ")[1].split(" ")[0] + " contains " }])
            pairs["prompt"] = pairs["prompt"].apply(lambda m: tokenizer.apply_chat_template(m, tokenize=False).rpartition(end_of_turn_token)[0])
        print("Example prompt: ", pairs["prompt"].iloc[0])
        NUM_MASKS = 15

    if args.task == "sudoku":
        pairs1 = pd.read_json("./datasets/sudoku4x4_200_per_empty_cell1to12.jsonl", orient='records', lines=True)
        backslash= "\n"
        # leading 0s in the Sudoku are dicarded by pandas --> add back in
        pairs1["converted_puzzle"] = pairs1["converted_puzzle"].apply(lambda x: (16-len(str(x)))*"0"+str(x))
        sysprompt = {"role": "system", "content": df["game_rule"].iloc[0] + "\nOnly provide the solved sudoku grid as a string of digits. Do not provide any additional explanation or text."}
        pairs1["prompt"] = pairs1.apply(lambda x:  [sysprompt, {"role": "user", "content": f"Solve the following Sudoku puzzle:\n0000\n0040\n4312\n0200"}, {"role": "assistant", "content": f"3421\n2143\n4312\n1234"},
                {"role": "user", "content": f"Solve the following Sudoku puzzle:\n0400\n3014\n2300\n4032"}, {"role": "assistant", "content": f"1423\n3214\n2341\n4132"},
                {"role": "user", "content": f"Solve the following Sudoku puzzle:\n{backslash.join([str(x['converted_puzzle'])[4*i:(4*i+4)] for i in range(4)])}"},
                {"role": "user", "content": f"{backslash.join([str(x['converted_puzzle'])[4*i:(4*i+4)] for i in range(4)])}".replace("0", tokenizer.mask_token)}], axis=1)
        pairs1["prompt"] = pairs1["prompt"].apply(lambda m: tokenizer.apply_chat_template(m, tokenize=False).rpartition(end_of_turn_token)[0])
        pairs1["sentence_id"] = pairs1.index

        # pairs2 = pairs1 shifted by one
        pairs2 = pairs1.iloc[1:].copy()
        pairs2.reset_index(drop=True, inplace=True)
        pairs2.loc[len(pairs2.index)] = pairs1.iloc[0].copy()

        pairs2["sentence_id"] = pairs2.index
        print("Example corr: ", pairs1["prompt"].iloc[0])
        print("Example clean: ", pairs2["prompt"].iloc[0])
        NUM_MASKS = 0

    for NUM_EOS in [8‚16, 32 ]:
            print("eos ", NUM_EOS)
            modelname = modelname.replace('/', '_')
            outfile = f"../activation_patching_results/{modelname}_{args.task}_allLayers_masks{NUM_MASKS}_eos{NUM_EOS}.csv"
            print("outfile ", outfile)
            results = pd.DataFrame(columns=["sentence_id", "layer", "clean_input", "corrupted_input", "clean_output", "corrupted_output", "patched_output"])
            
            # tokenize and save index of eos tokens <- a pair should have equally many eos tokens
            for (_, clean_row), (_, corrupted_row) in zip(pairs1.iterrows(), pairs2.iterrows()):

                clean_tok = tokenizer(clean_row["prompt"], add_special_tokens=False)["input_ids"]
                length_clean = len(clean_tok)

                corrupted_tok = tokenizer(corrupted_row["prompt"], add_special_tokens=False)["input_ids"]
                length_corrupt = len(corrupted_tok)
                diff = length_clean-length_corrupt

                clean_tok.extend([tokenizer.mask_token_id]*(NUM_MASKS + max(-diff, 0)) +  [tokenizer.eos_token_id]*NUM_EOS)
                length_clean_with_eos = len(clean_tok)
                clean_tok = torch.tensor(clean_tok).unsqueeze(0)

                corrupted_tok.extend([tokenizer.mask_token_id]*(NUM_MASKS + max(diff, 0)) + [tokenizer.eos_token_id]*NUM_EOS)
                length_corrupt_with_eos = len(corrupted_tok)
                corrupted_tok = torch.tensor(corrupted_tok).unsqueeze(0)

                # generate output logits
                if "2.0" in modelname:
                    position_ids = torch.arange(length_corrupt_with_eos, device="cuda").unsqueeze(0)
                    output = model(corrupted_tok.to("cuda"), attention_mask=torch.ones(corrupted_tok.shape).to("cuda"), position_ids=position_ids, output_hidden_states=True, return_dict=True)
                else:
                    output = model(corrupted_tok.to("cuda"))
                
                # sample from logits
                if "1.5" in modelname:
                    logits_with_noise = add_gumbel_noise(output.logits[0], temperature=0.0)
                    corrupted_gen = torch.argmax(logits_with_noise, dim=-1)
                elif "dream" in modelname.lower():
                    _ ,corrupted_gen = sample_tokens(output.logits[0])
                elif "2.0" in modelname.lower():
                    corrupted_gen, _ = model._sample_with_temperature_topk_topp(output.logits[0])
                    
                corrupted_gen = tokenizer.decode(corrupted_gen)

                # save the hidden states for patching
                hs_corrupted = output.hidden_states
                
                
                for layer in range(0, numlayers, 2):
                    if "1.5" in modelname:
                        layer_name = f"model.transformer.blocks.{layer}"
                    elif "dream" in modelname.lower() or "2.0" in modelname:
                        layer_name = f"model.layers.{layer}"
                    target_layer = dict(model.named_modules()).get(layer_name, None)

                    if target_layer is None:
                        print(f"Layer {layer_name} not found in the model.")

                    # get the hidden states of the eos tokens
                    eos_corrupted = hs_corrupted[layer][:,:-NUM_EOS,:]

                    def patch_fn(activations):
                        # replace the activations of the eos tokens
                        activations[0][:, :-NUM_EOS, :] = eos_corrupted
                        return activations

                    def hook(module, input, output):
                        patched = patch_fn(output)
                        return patched

                    # run an uncorrupted forward pass with the original input
                    if "2.0" in modelname:
                        position_ids = torch.arange(length_clean_with_eos, device="cuda").unsqueeze(0)
                        output = model(clean_tok.to("cuda"), attention_mask=torch.ones(clean_tok.shape).to("cuda"), position_ids=position_ids)
                    else:
                        output = model(clean_tok.to("cuda"))
                    if "1.5" in modelname:
                        logits_with_noise = add_gumbel_noise(output.logits[0], temperature=0.0)
                        clean_gen = torch.argmax(logits_with_noise, dim=-1)
                    elif "dream" in modelname.lower():
                        _ ,clean_gen = sample_tokens(output.logits[0])
                    elif "2.0" in modelname.lower():
                        clean_gen, _ = model._sample_with_temperature_topk_topp(output.logits[0])
                    clean_gen =  tokenizer.decode(clean_gen)

                    # register hook and run a patched forward pass
                    handle = target_layer.register_forward_hook(hook)
                    if "2.0" in modelname:
                        position_ids = torch.arange(length_clean_with_eos, device="cuda").unsqueeze(0)
                        out_patched = model(clean_tok.to("cuda"), attention_mask=torch.tensor(length_clean_with_eos*[1.0]).unsqueeze(0).to("cuda"), position_ids=position_ids)
                    else:
                        out_patched = model(clean_tok.to("cuda"))

                    if "1.5" in modelname:
                        logits_with_noise = add_gumbel_noise(out_patched.logits[0], temperature=0.0)
                        patched_gen = torch.argmax(logits_with_noise, dim=-1)
                    elif "dream" in modelname.lower():
                        _ ,patched_gen = sample_tokens(out_patched.logits[0])
                    elif "2.0" in modelname.lower():
                        patched_gen, _ = model._sample_with_temperature_topk_topp(out_patched.logits[0])
                    patched_gen = tokenizer.decode(patched_gen)

                    handle.remove()

                    # write results to file
                    with open(outfile, "a", newline='') as f:
                        tsvwriter = csv.writer(f, delimiter='\t')
                        tsvwriter.writerow([clean_row["sentence_id"], layer, tokenizer.decode(clean_tok[0]), tokenizer.decode(corrupted_tok[0]), clean_gen, corrupted_gen, patched_gen])
                    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Select the task and the model.")
    parser.add_argument('--task', choices=['maths', 'boxes', 'sudoku'], type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    print("Running ", args.task)
    main(args.task, args.model)