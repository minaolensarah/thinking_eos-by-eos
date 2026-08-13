from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM 
import pandas as pd
from torch.utils.data import DataLoader
import torch
import argparse, csv, os, sys
from datasets import Dataset
from my_modeling_llada2_moe import My_LLaDA2MoeModelLM
from generate_llada1.5 import generate, generate_fix_eos
import random


def main():
    parser = argparse.ArgumentParser(description="Evaluation / inference arguments")
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1,
                help='batch_size')
    # generation
    parser.add_argument('--steps', dest='steps', type=int, default=None,
                help='Number of Remasking steps')
    parser.add_argument('--num_eos', dest='num_eos', type=int, default=0,
                help='Number of prefilled eos tokens')
    parser.add_argument('--num_masks', dest='num_masks', type=int, default=32,
                help='generation length apart from prefilled eos tokens')
    parser.add_argument('--remasking', dest='remasking', type=str, default="low_confidence",
                help='Remasking strategy')
    parser.add_argument('--block_length', dest='block_length', type=int, default=None,
                help='Semi autoregressive blocks, if blocklen == gen length, generation is not semi autoregressive')
    parser.add_argument('--gen_length', dest='gen_length', type=int, default=8,
                help='Generation length, if blocklen == gen length, generation is not semi autoregressive')
    parser.add_argument('--temp01',
                        dest="temp01",
                        action="store_true")

    # paths
    parser.add_argument('--data_path', dest='data_path', type=str, help='Path to evaluation data file (jsonl)')
    parser.add_argument('--name', dest='name', type=str, default="test")
    parser.add_argument('--model', dest='model', type=str, default="GSAI-ML/LLaDA-1.5",
                help='Path to model checkpoint or model identifier from huggingface.co/models')
    parser.add_argument('--tokenizer', dest='tokenizer', type=str, default=None,
                help='Path to tokenizer otherwise the model tokenizer will be used')
    parser.add_argument('--outfolder', dest='outfolder', type=str, default="./outfiles",
                help='Where to save the output')

    # alternative padding tokens
    parser.add_argument('--seed', dest='seed', type=int, default=42,
                help='Random seed for random pad tokens')
    parser.add_argument('--random_tokens',
                        dest="random_tokens",
                        action="store_true")
    parser.add_argument('--eos_id', dest='eos_id', type=int, default=None,
                help="ID of the eos token, leave None to use the tokenizer's EoS token")
    
    # tasks
    parser.add_argument('--boxes',
                        dest="boxes",
                        action="store_true")
    parser.add_argument('--sudoku',
                        dest="sudoku",
                        action="store_true")
    parser.add_argument('--maths',
                        dest="maths",
                        action="store_true")
    parser.add_argument('--gsm8k',
                        dest="gsm8k",
                        action="store_true")
    parser.add_argument('--twohop',
                        dest="twohop",
                        action="store_true")




    parser.add_argument('--cot',
                        dest="cot",
                        action="store_true")

    # for gsm8k
    parser.add_argument('--shots', dest='shots', type=int, default=0,
                help='Number of shots for few-shot learning')
    parser.add_argument('--detailleddemo',
                        dest="detailleddemo",
                        action="store_true")

    # use for llada sudoku
    parser.add_argument('--eot',
                        dest="eot",
                        action="store_true")
            


    args, _ = parser.parse_known_args()

    random.seed(args.seed)
    current_state = random.getstate()

    modelname = args.model

    filename = args.data_path.split('/')[-1]
    if args.block_length is None:
        args.block_length = args.gen_length

    if args.steps is None:
        args.steps = args.gen_length

    if args.random_tokens:
        args.eos_id = -1
        print("Padding with random tokens")
    
    if args.num_eos > 0:
        steps = f"numeos{args.num_eos}_masks{args.num_masks}_steps{args.steps}_{args.remasking}{'_padwith' + str(args.eos_id) if args.eos_id is not None else ''}"
        if args.eot:
            steps+= "Eot"
    modelname = args.model.replace("/", "_")
    if args.cot:
        args.name += "_cot"
    outfile = f"{args.outfolder}/{modelname}_{args.name}_{filename}_results_{steps}.tsv"

    print(modelname)
    if os.path.exists(outfile):
        try:
            out = pd.read_csv(outfile, sep="\t", header=None)
            completed = len(out.index)
        except pd.errors.EmptyDataError:
            completed = 0
    else:
        completed = 0
    print(outfile)
    print("Starting from sample number: ", completed)
    if args.tokenizer is None:
        print(args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
            trust_remote_code=True,
        )
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            use_fast=True,
            trust_remote_code=True,
        )
    
    # if no potential reasoning token is specified, use eos token
    if args.eos_id is None:
        args.eos_id = tokenizer.eos_token_id
    
    if "llada" in modelname.lower():
        tokenizer.padding_side = 'left'
        assert tokenizer.pad_token_id != 126336
    else:
        tokenizer.pad_token = tokenizer.eos_token 
    if "dream" in modelname.lower():    
        tokenizer.padding_side = 'left'

    if "jsonl" in args.data_path:
        data = pd.read_json(args.data_path, orient='records', lines=True)
    else:
        data = pd.read_csv(args.data_path, sep="\t")

    if completed == len(data.index):
        sys.exit()
        print("File already done")
    
    if "llama" in modelname.lower() or "qwen" in modelname.lower() or ("llada2" in modelname.lower() and args.num_eos == 0):
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype="auto", device_map="auto")
    elif "llada2" in modelname.lower() and args.num_eos > 0:
        model = My_LLaDA2MoeModelLM.from_pretrained(args.model, trust_remote_code=True, dtype="auto", device_map="auto")
    else:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype="auto", device_map="auto")

    while completed < len(data.index) and args.batch_size >0:
        df = data.iloc[completed:].copy()
        sysprompt = {"role": "system", "content":"Answer the question but do not give any additional explanation."}
        if args.cot:
            sysprompt = {"role": "system", "content":"Reason step-by-step and explain your thoughts."}
        if args.boxes:
            df["sentence_masked"] = df["sentence_masked"].apply(lambda x: x.replace("<extra_id_0> .", ""))
            if args.cot:
                sysprompt = {"role": "system", "content":"Reason step-by-step and explain your thoughts. Then answer the question with 'Box <number> contains <content>.'"}
            else:
                df["prompt"] = df["sentence_masked"].apply(lambda sample: [sysprompt, {"role": "user", "content": sample.rpartition(".")[0] + "." + "\nWhat does Box " + sample.rpartition(".")[2].split("Box ")[1].split(" ")[0] + " contain?" + fill}])
        elif args.sudoku:
            df["converted_puzzle"] = df["converted_puzzle"].apply(lambda x: (16-len(str(x)))*"0"+str(x))
            sysprompt = {"role": "system", "content": df["game_rule"].iloc[0] + "\nOnly provide the solved sudoku grid as a string of digits. Do not provide any additional explanation or text."}
            if args.cot:
                sysprompt = {"role": "system", "content": df["game_rule"].iloc[0] + "\nReason step-by-step and explain your thoughts. Finally write the completed Sudoku is <final_result>."}
            df["prompt"] = df.apply(lambda x:  [sysprompt, {"role": "user", "content": f"Solve the following Sudoku puzzle:\n0000\n0040\n4312\n0200"}, {"role": "assistant", "content": f"3421\n2143\n4312\n1234"},
            {"role": "user", "content": f"Solve the following Sudoku puzzle:\n0400\n3014\n2300\n4032"}, {"role": "assistant", "content": f"1423\n3214\n2341\n4132"},
             {"role": "user", "content": f"Solve the following Sudoku puzzle:\n{'\n'.join([str(x['converted_puzzle'])[4*i:(4*i+4)] for i in range(4)])}"}], axis=1)
        elif args.maths:
            sysprompt = {"role": "system", "content":"Answer the question only with the number that is the final result. Do not give any additional explanation."}
            if args.cot:
                sysprompt = {"role": "system", "content":"Reason step-by-step and explain your thoughts. Finally write 'The final result is ...'."}
            df["prompt"] = df.apply(lambda x:  [sysprompt, {"role": "user", "content": f"What is the result of {x['calculation']}?"}], axis=1)
        elif args.twohop:
            if not args.cot:
                sysprompt = {"role": "system", "content":"Provide only the name of the entity described by the statement. Do not give any explanation or reasoning trace."}
                df["prompt"] = df.apply(lambda x:  [sysprompt, {"role": "user", "content": x['source_prompt'].capitalize()}], axis=1)
            else:
                sysprompt = {"role": "system", "content":"Answer the question with the name of the entity. Do not write any explanation or reasoning trace."}
                df["prompt"] = df.apply(lambda x:  [sysprompt, {"role": "user", "content": f"Provide the name of {x['source_prompt'][:-3] if x['source_prompt'][-3:] == ' is' else x['source_prompt']}."}], axis=1)
        elif args.gsm8k:
            if args.cot:
                sysprompt = {"role": "system", "content": "Given a problem, reason and give a final answer to the problem.\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."}
            else:
                sysprompt = {"role": "system", "content": "Given the following problem, give a final answer to the problem. Give only the result as an answer. Do not write any further explanations. Your response should be \"The final answer is [answer]\" where [answer] is the response to the problem."}
            demo = []
            examples = ""
            reminder =""
            if args.shots > 0:
                shots = pd.read_csv("datasets/gsm8k_fewshot_full.tsv", sep="\t")
                reminder = "\nRemeber, do not write down your reasoning, only give the answer."
                examples = "Here are some examples of problems and how to arrive at the final answer. In your answer only give the answer and not the reasoning.\n"
                for i in range(args.shots):
                    examples += f"Problem: {shots['question'].iloc[i]}\n"
                    examples += f"Reasoning: {shots['target'].iloc[i].split('The final')[0]}\n"
                    examples += f"Answer: The final answer is {shots['target'].iloc[i].split('The final answer is ')[1]}\n"
                examples += "\n"
                df["prompt"] = df.apply(lambda x:  [sysprompt] + demo + [{"role": "user", "content":examples + "Problem: " + x["question"] + reminder}], axis=1)
        
        else:
            print("no setup specified")
            
        if "qwen" in modelname.lower():
            df["prompt"] = df["prompt"].apply(lambda m: tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking = args.cot))
        else:
            df["prompt"] = df["prompt"].apply(lambda m: tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True))

        print("Example prompt: ", df["prompt"].iloc[0])

        dataset = Dataset.from_pandas(df[["prompt"]])
        def tokenize(examples):
            return tokenizer(examples["prompt"], return_tensors="pt", padding="longest", add_special_tokens=False)
        dataset = dataset.map(lambda e: tokenize(e), batched=True, batch_size=args.batch_size)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        dataset = DataLoader(dataset, batch_size=args.batch_size)

        
        print("Starting generation...")
        for sample in dataset: 
            
            if "llada2" in modelname.lower():
                input_ids = sample['input_ids'].to("cuda")
                attention_mask = sample['attention_mask'].to("cuda")
                if args.num_eos > 0:
                    out = model.generate(input_ids, 
                            steps=args.steps, 
                            num_masks=args.num_masks, 
                            num_eos=args.num_eos, 
                            temperature=0.0 if not args.temp01 else 0.1, 
                            dont_add_eot=(not args.eot))
                else:
                    out = model.generate(
                            inputs=input_ids,
                            eos_early_stop=args.cot,
                            gen_length=args.gen_length,
                            block_length=args.block_length,
                            steps=args.steps,
                            temperature=0.0 if not args.temp01 else 0.1,
                        )
                                            
                output_text = tokenizer.batch_decode(out)

            elif "llada" in modelname.lower(): 
                input_ids = sample['input_ids'].to("cuda")
                attention_mask = sample['attention_mask'].to("cuda")
                if args.num_eos >0:
                    
                    out, current_state = generate_fix_eos(model, 
                    input_ids, attention_mask, steps=args.steps, num_masks=args.num_masks, 
                    num_eos=args.num_eos, block_length=args.block_length, 
                    temperature=0.0 if not args.temp01 else 0.1, cfg_scale=0., 
                    remasking=args.remasking, dont_add_eot=(not args.eot), 
                    eos_id=args.eos_id, random_state=current_state)
                    
                    output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
                else:
                    out = generate(model, input_ids, attention_mask, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=0.0 if not args.temp01 else 0.1, cfg_scale=0., remasking=args.remasking, confidence_eos_eot_inf=args.confidence_eos_eot_inf)
                    output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
            elif "llama" in modelname.lower() or "qwen" in modelname.lower():
                input_ids = sample['input_ids'].to("cuda")
                attention_mask = sample['attention_mask'].to("cuda")
                out = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.gen_length) #, temperature=0.0)
                out.to("cpu")
                output_text = tokenizer.batch_decode(out, skip_special_tokens=True)
            elif "dream" in modelname.lower():
                input_ids = sample['input_ids'].to("cuda")
                attention_mask = sample['attention_mask'].to("cuda")
                if args.num_eos >0:

                    def generation_tokens_hook_func(a,tokens , b):
                        if args.random_tokens:
                            tokens[:, -args.num_eos:] = torch.LongTensor([random.choices(list(range(151667)), k=args.num_eos) for i in range(tokens.shape[0])]).to(model.device)
                        else:
                            tokens[:, -args.num_eos:] = torch.full((tokens.shape[0], args.num_eos), args.eos_id, dtype=torch.long).to(model.device)
                        return tokens

                    out = model.diffusion_generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.num_eos + args.num_masks, 
                        output_history=False,
                        return_dict_in_generate=True,
                        generation_tokens_hook_func=generation_tokens_hook_func,
                        steps=args.steps,
                        temperature=0.0 if not args.temp01 else 0.1,
                        alg="entropy",
                        alg_temp=0.,
                    )

                else: 
                    out = model.diffusion_generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.gen_length,
                        output_history=False,
                        return_dict_in_generate=True,
                        steps=args.steps,
                        temperature=0.0 if not args.temp01 else 0.1,
                        #top_p=0.95,
                        alg="entropy",
                        alg_temp=0.,
                    )
                output_text = tokenizer.batch_decode(out.sequences[:, input_ids.shape[1]:], skip_special_tokens=True)
            else:
                print("which model?")

                        
            with open(outfile, "a", newline='') as f:
                tsvwriter = csv.writer(f, delimiter='\t')
                for o in output_text:
                    tsvwriter.writerow([o, df["prompt"].loc[completed]] + ([df["id"].loc[completed]] if "id" in df.columns else []))
                    completed +=1
        

if __name__ == "__main__":
    main()