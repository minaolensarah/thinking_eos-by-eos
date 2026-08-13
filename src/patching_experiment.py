
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM #ForCausalLM
import torch 
import pandas as pd
import torch.nn as nn
from collections import Counter
import csv
import random
import sys

import argparse

parser = argparse.ArgumentParser(description="Select the task.")
parser.add_argument('--task', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--pad_with', type=str)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--name', default='')

args = parser.parse_args()

random.seed(args.seed)

print("Running ", args.task)


def get_avg_rank(logits, token_ids, input_ids, mask):
    """
    Get the average rank in the logits predicted by the model of the tokens (token_ids) at the positions that are masked in the input.
    """
    avg = 0
    total = 0
    for ix, (tok_id, inputid) in enumerate(zip(token_ids, input_ids)):
        if inputid == mask:
            total += 1
            logit = float(logits[ix][tok_id])
            rank = int((logits[ix] > logit).sum()) + 1
            avg += rank
    return avg/total

def get_hook(curr_layer, numeos, hs_corrupted):
    """
    Hook wrapper for patching activations at a specific layer.
    """
    def patch_fn(activations):
        activations[0][:, -numeos:, :] = hs_corrupted[curr_layer+1][:,-numeos:,:]
        return activations

    def hook(module, args, output):
        patched = patch_fn(output)
        return patched
    return hook

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
    


if "2.0" in args.model:
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, device_map="auto")
else:
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, device_map="auto")

model.tie_weights()

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)

if "1.5" in args.model:
    tokenizer.mask_token_id = 126336
    end_of_turn_token = "<|eot_id|>"
elif "dream" in args.model.lower():
    end_of_turn_token = "<|im_end|>"
elif "2.0" in args.model:
    end_of_turn_token = "<|role_end|>"

print("mask token id:", tokenizer.mask_token_id)
print("Padding token:", args.pad_with)
numlayers = model.config.num_hidden_layers
print("number of layers:", numlayers)

if args.task == "maths":
    pairs1 = pd.read_json("datasets/easy_maths_200perCalc.jsonl", orient='records', lines=True)
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
    sysprompt = {"role": "system", "content":"Answer the question only with the number that is the final result. Do not give any additional explanation."}
    for df in [pairs1, pairs2]:
        df["sentence_id"] = df.index
        df["prompt"] = df.apply(lambda x:  [sysprompt, {"role": "user", "content": f"What is the result of {x['calculation']}?"}], axis=1)
        df["prompt"] = df["prompt"].apply(lambda m: tokenizer.apply_chat_template(m, tokenize=False))#.rpartition(end_of_turn_token)[0])
    print("Example prompt: ", pairs1["prompt"].iloc[0])
    print("Example prompt: ", pairs2["prompt"].iloc[0])
    NUM_MASKS = 4

if args.task == "boxes":
    pairs1 = pd.read_json("datasets/boxes_testset_24X30_pair1.jsonl", orient='records', lines=True)
    pairs2 = pd.read_json("datasets/boxes_testset_24X30_pair2.jsonl", orient='records', lines=True)
    
    for pairs in [pairs1, pairs2]:
        pairs["sentence_masked"] = pairs["sentence_masked"].apply(lambda s: s.replace("<extra_id_0> .", ""))
        sysprompt = {"role": "system", "content":"Answer the question but do not give any additional explanation."}
        pairs["prompt"] = pairs["sentence_masked"].apply(lambda sample: [sysprompt, {"role": "user", "content": sample.rpartition(".")[0] + "." + "\nWhat does Box " + sample.rpartition(".")[2].split("Box ")[1].split(" ")[0] + " contain?" }])
        pairs["prompt"] = pairs["prompt"].apply(lambda m: tokenizer.apply_chat_template(m, tokenize=False))
    
    print("Example prompt: ", pairs1["prompt"].iloc[0])
    print("Example prompt: ", pairs2["prompt"].iloc[0])

    NUM_MASKS = 22 

if args.task == "sudoku":
    def make_parallel_sudoku(sudoku):
        counterfactual = ""
        mapping = ["0", "2", "3", "4", "1"]
        for s in sudoku:
            counterfactual += mapping[int(s)]
        return counterfactual 

    pairs1 = pd.read_json("datasets/sudoku4x4_200_per_empty_cell1to12.jsonl", orient='records', lines=True)
    backslash= "\n"
    pairs1["converted_puzzle"] = pairs1["converted_puzzle"].apply(lambda x: (16-len(str(x)))*"0"+str(x))
    
    pairs2 = pairs1.copy()
    pairs2["converted_puzzle"] = pairs2["converted_puzzle"].apply(make_parallel_sudoku)

    for df in [pairs1, pairs2]:
        sysprompt = {"role": "system", "content": df["game_rule"].iloc[0] + "\nOnly provide the solved sudoku grid as a string of digits. Do not provide any additional explanation or text."}
        df["prompt"] = df.apply(lambda x:  [sysprompt, {"role": "user", "content": f"Solve the following Sudoku puzzle:\n0000\n0040\n4312\n0200"}, {"role": "assistant", "content": f"3421\n2143\n4312\n1234"},
                {"role": "user", "content": f"Solve the following Sudoku puzzle:\n0400\n3014\n2300\n4032"}, {"role": "assistant", "content": f"1423\n3214\n2341\n4132"},
                {"role": "user", "content": f"Solve the following Sudoku puzzle:\n{backslash.join([str(x['converted_puzzle'])[4*i:(4*i+4)] for i in range(4)])}"},
                {"role": "user", "content": f"{backslash.join([str(x['converted_puzzle'])[4*i:(4*i+4)] for i in range(4)])}".replace("0", tokenizer.mask_token)}], axis=1)
        df["prompt"] = df["prompt"].apply(lambda m: tokenizer.apply_chat_template(m, tokenize=False).rpartition(end_of_turn_token)[0])
        df["sentence_id"] = df.index

    print("Example corr: ", pairs1["prompt"].iloc[0])
    print("Example clean: ", pairs2["prompt"].iloc[0])
    NUM_MASKS = 0


for NUM_EOS in [64 ]: #‚16, 32, 128 ]: paper with numeos 54‚
        print("eos ", NUM_EOS)
        modelname = args.model.replace('/', '_')
        outfile = f"activation_patching_results/{modelname}_{args.task}{args.name}_logits_masks{NUM_MASKS}_eos{NUM_EOS}{args.pad_with}_seed{args.seed}.csv"
        try:
            done = pd.read_csv(outfile, sep="\t",names=["sentence_id", "layer", "clean_input", "corrupted_input", "clean_output", "corrupted_output", "patched_output", "avg_counterf_in_clean", "avg_counterf_in_patched", "avg_clean_in_patched"])
            done_datapoints = (set(done["sentence_id"]))
            if int(done.iloc[-1]["layer"]) < numlayers-1:
                last_datapoint = done.iloc[-1]["sentence_id"]
                done_datapoints.remove(last_datapoint)
                done = done[done["sentence_id"]!= last_datapoint]
                done.to_csv(outfile, sep="\t", index=False, header=False)
            
            max_layer = 0
            pairs1 = pairs1[pairs1["sentence_id"].isin(done_datapoints)==False]
            pairs2 = pairs2[pairs2["sentence_id"].isin(done_datapoints)==False]
            print("resuming from", len(pairs1.index))

        except FileNotFoundError:
            max_layer = 0
        print("outfile ", outfile)
        
        with torch.no_grad():
            for (_, clean_row), (_, corrupted_row) in zip(pairs1.iterrows(), pairs2.iterrows()):
                all_hooks= []

                clean_tok = tokenizer(clean_row["prompt"], add_special_tokens=False)["input_ids"]
                length_clean = len(clean_tok)

                corrupted_tok = tokenizer(corrupted_row["prompt"], add_special_tokens=False)["input_ids"]
                length_corrupt = len(corrupted_tok)

                diff = length_clean-length_corrupt
                assert length_clean == length_corrupt, "The two prompts have a different length"
                
                padding = []
                if args.pad_with == "eos":
                    padding = [tokenizer.eos_token_id] *NUM_EOS 
                elif args.pad_with == "dots":
                    padding = [13] *NUM_EOS 
                elif args.pad_with == "random":
                    if "1.5" in args.model:
                        padding = random.choices(list(range(tokenizer.vocab_size-1)),k=NUM_EOS) 
                    else:   
                        padding = random.choices(list(range(tokenizer.total_vocab_size-1)),k=NUM_EOS)
                elif args.pad_with == "whitespace":
                    padding = [220] *NUM_EOS 
                else:
                    print("No valid padding specified")
                    sys.exit()
                
                clean_tok.extend([tokenizer.mask_token_id]*(NUM_MASKS ) +  padding)

                length_clean_with_eos = len(clean_tok)
                clean_tok = torch.tensor(clean_tok).unsqueeze(0)

                corrupted_tok.extend([tokenizer.mask_token_id]*(NUM_MASKS ) + padding)
                length_corrupt_with_eos = len(corrupted_tok)
               
                corrupted_tok = torch.tensor(corrupted_tok).unsqueeze(0)

                output = model(clean_tok.to("cuda"))
                
                
                if "1.5" in args.model:
                    clean_logits = add_gumbel_noise(output.logits[0], temperature=0.0)
                    clean_token_ids = torch.argmax(clean_logits, dim=-1)
                elif "dream" in args.model.lower():
                    clean_logits = output.logits[0]
                    _ ,clean_token_ids = sample_tokens(clean_logits)
                
                clean_gen_decoded = tokenizer.decode(clean_token_ids)

                clean_logits = clean_logits.to("cpu")

                output = model(corrupted_tok.to("cuda"), output_hidden_states=True, return_dict=True)
                if "1.5" in args.model:
                    counterf_logits = add_gumbel_noise(output.logits[0], temperature=0.0)
                    corrupted_token_ids = torch.argmax(counterf_logits, dim=-1)
                elif "dream" in args.model.lower():
                    counterf_logits = output.logits[0]
                    _ ,corrupted_token_ids = sample_tokens(counterf_logits)

                corrupted_token_ids = corrupted_token_ids.to("cpu")
                avg_counterf_in_clean = get_avg_rank(clean_logits, corrupted_token_ids, clean_tok[0].to("cpu"), tokenizer.mask_token_id)
                

                corrupted_gen_decoded = tokenizer.decode(corrupted_token_ids)

                hs_corrupted = [h[:,-NUM_EOS:,:].detach().clone() for h in output.hidden_states] 

                for upuntil_layer in range(max_layer, numlayers, 2):
                    for layer in [ upuntil_layer-1, upuntil_layer]:
                        if layer <0:
                            continue
                        if "1.5" in args.model:
                            layer_name = f"model.transformer.blocks.{layer}"
                        elif "dream" in args.model.lower() or "2.0" in args.model:
                            layer_name = f"model.layers.{layer}"
                        target_layer = dict(model.named_modules()).get(layer_name, None)
                        if target_layer is None:
                            print(f"Layer {layer_name} not found in the model.")

                        # register hook and save handle
                        all_hooks.append(target_layer.register_forward_hook(get_hook(layer, NUM_EOS, hs_corrupted)))
                    
                    out_patched = model(clean_tok.to("cuda"))

                    if "1.5" in args.model:
                        patched_logits = add_gumbel_noise(out_patched.logits[0], temperature=0.0)
                        patched_gen = torch.argmax(patched_logits, dim=-1)
                    elif "dream" in args.model.lower():
                        _ ,patched_gen = sample_tokens(out_patched.logits[0])
                        patched_logits = out_patched.logits[0]
                    
                    patched_gen = tokenizer.decode(patched_gen)

                    patched_logits = patched_logits.to("cpu")
                    corrupted_token_ids = corrupted_token_ids.to("cpu")
                    avg_counterf_in_patched = get_avg_rank(patched_logits, corrupted_token_ids, clean_tok[0], tokenizer.mask_token_id)
                    avg_clean_in_patched = get_avg_rank(patched_logits, clean_token_ids, clean_tok[0], tokenizer.mask_token_id)

                    with open(outfile, "a", newline='') as f:
                        tsvwriter = csv.writer(f, delimiter='\t')
                        tsvwriter.writerow([clean_row["sentence_id"], layer, tokenizer.decode(clean_tok[0]), tokenizer.decode(corrupted_tok[0]), clean_gen_decoded, corrupted_gen_decoded, patched_gen, avg_counterf_in_clean, avg_counterf_in_patched, avg_clean_in_patched])
                    
                max_layer = 0
                for handle in all_hooks:
                    handle.remove()
                