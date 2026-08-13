import pandas as pd 
import argparse
import re
import seaborn as sns
import os
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from datasets import load_dataset
from unidecode import unidecode

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def preprocess_predicition(x):
    x = str(x)
    x = x.replace("<|role_end|><|endoftext|>", "")

    x = x.replace("USA", "united states of america")
    x = x.replace("Washington", "Washington, D.C.")
    x = x.replace("US Dollar", "United States dollar")
    x = x.replace("Saint ", "St. ")
    x = x.replace("British Pound", "pound sterling")
    if x.strip().lower() == "america":
        x = "united states of america"

    if x.strip().lower() == "united states" or x.strip().lower() == "united states america":
        x = "united states of america"
    elif x.strip().lower() == "british":
        x="united kingdom"
    elif x.strip().lower() == "england":
        x="united kingdom"
    elif "princess grace of monaco" in x.lower() :
        x="Grace Kelly"
    return x.lower()

testset = pd.read_csv("./datasets/two_hop_answerlenmax5_fullcolumns.csv", sep="\t")
testset["prompt_e1"] = testset.apply(lambda x: x["r1_template"].replace("{}", x["e1_label"]), axis = 1)
testset["prompt_e2"] = testset.apply(lambda x: x["r2_template"].replace("{}", x["e2_label"]), axis = 1)
print("Testset Length ", len(testset.index))

answers_e1 = pd.read_csv("./datasets/one_hop_e1_answerlenmax5.csv", sep="\t")
answers_e2 = pd.read_csv("./datasets/one_hop_e2_answerlenmax5.csv", sep="\t")

# evaluate the one hop facts 
for model in ["meta-llama_Llama-3.1-8B-Instruct", "Qwen_Qwen3-8B", "Dream-org_Dream-v0-Instruct-7B", "GSAI-ML_LLaDA-1.5","inclusionAI_LLaDA2.0-mini"]: 
    hop_e1path = f"./outfiles/exp4/{model}_test_one_hop_e1_answerlenmax5.csv_0shot_results_{'genlength10_steps10_blocklen10' if 'D' in model else ''}_temp01_seed42.tsv"
    hop_e2path = f"./outfiles/exp4/{model}_test_one_hop_e2_answerlenmax5.csv_0shot_results_{'genlength10_steps10_blocklen10' if 'D' in model else ''}_temp01_seed42.tsv"

    hop_e1 = pd.read_csv(hop_e1path, sep="\t", names=["prediction", "prompt", "id"])
    hop_e1["target"] = answers_e1["e2_label"]
    hop_e1["source_prompt"] = answers_e1["source_prompt"]

    hop_e1["prepo"] = hop_e1["prediction"].apply(preprocess_predicition)
    hop_e1["eval"] = hop_e1.apply(lambda x: (str(x["target"]).lower() in x["prepo"]) or (unidecode(str(x["target"]).lower()) in unidecode(x["prepo"])), axis=1)

    hop_e1[["prediction", "target", "eval"]].to_csv(hop_e1path + "eval.tsv", sep="\t", index=False)

    hop_e2 = pd.read_csv(hop_e2path, sep="\t", names=["prediction", "prompt", "id"])
    hop_e2["target"] = answers_e2["e3_label"]
    hop_e2["source_prompt"] = answers_e2["source_prompt"]

    hop_e2["prepo"] = hop_e2["prediction"].apply(preprocess_predicition)
    hop_e2["eval"] = hop_e2.apply(lambda x: (str(x["target"]).lower() in x["prepo"]) or (unidecode(str(x["target"]).lower()) in unidecode(x["prepo"])), axis=1)

    hop_e2[["prediction", "target", "eval"]].to_csv(hop_e2path + "eval.tsv", sep="\t", index=False)

    def get_included(row):
        if (hop_e1[hop_e1["source_prompt"]==row["prompt_e1"]]["eval"].iloc[0] == True) and (hop_e2[hop_e2["source_prompt"]==row["prompt_e2"]]["eval"].iloc[0] == True):
            return True
        else:
            return False
    testset[model + "_subset"] = testset.apply(get_included, axis=1)


    print(model, testset[model + "_subset"].sum(), testset[model + "_subset"].mean())


# evaluate the two hop data, only consider datapoints where the two individual hops were correct
accs_tables = pd.DataFrame(columns=["model", "numeos", "eval"])
for file in os.scandir("./outfiles/exp4"):
    if "eval" in file.name or "gsm8k" in file.name or "one_hop" in file.name or "temp00" in file.name:
        continue
    results = pd.read_csv(file, sep="\t", names=["prediction", "prompt", "id"])
    print("Length ", len(results.index))
    model = "_".join(file.name.split("_")[:2]) 
    results["target"] = testset["e3_label"]
    results[model+"_subset"] = testset[model+"_subset"]

    results["model"] = model
    results["numeos"] = int(file.name.split("numeos")[1].split("_")[0]) if "numeos" in file.name else 1

    results["prepo"] = results["prediction"].apply(preprocess_predicition)
    results["eval"] = results.apply(lambda x: (str(x["target"]).lower() in x["prepo"]) or (unidecode(str(x["target"]).lower()) in unidecode(x["prepo"])), axis=1)
    
    results[["prediction", "target", "eval"]].to_csv(file.path + "eval.tsv", sep="\t", index=False)

    print(f"File: {file.name} - Accuracy: {results[results[model+'_subset']==True]['eval'].mean()} - Length {len(results[results[model+'_subset']==True])}")
    accs_tables = pd.concat([accs_tables, results[results[model+'_subset']==True][["model", "numeos", "eval"]]], axis=0)


accs_tables.to_csv("./twohop_all_accs.tsv", sep="\t", index=False)

model_rename = {'LLaDA-1.5':'LLaDA-1.5', 'Dream-v0-Instruct-7B':"Dream-v0", 'LLaDA2.0-mini':"LLaDA2.0-mini", 'Qwen3-8B':'Qwen3', 'Llama-3.1-8B-Instruct': 'Llama-3.1'}
accs_tables["model"] = accs_tables["model"].apply(lambda x: model_rename[x.split("_")[1]])
g = sns.relplot(accs_tables, x="numeos", y="eval", kind="line", hue="model", 
    errorbar="se",
    hue_order=['Dream-v0', 'LLaDA-1.5', 'LLaDA2.0-mini'], 
    palette={'LLaDA-1.5':"tab:green", 'Dream-v0':"tab:blue", 'LLaDA2.0-mini':"tab:purple"},)

plt.ylim(0,0.65)
for ax in g.axes.flat:
    ax.set_xscale("log", base=2)
g.set(xticks=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], xticklabels=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])#, base=2)
g._legend.set_title("Models")

plt.title("Two hop")
plt.ylabel("Accuracy")
plt.xlabel("EoS Tokens")
plt.xticks(fontsize=10)
plt.savefig("./twohop_accsOverEos.pdf", format="pdf")

print("done")