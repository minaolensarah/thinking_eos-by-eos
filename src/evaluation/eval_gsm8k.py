import pandas as pd 
import argparse
import re
import seaborn as sns
import os
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


testset = pd.read_json("./datasets/test-canonical_gsm8k.jsonl", lines=True, orient="records")
print("Testset Length ", len(testset.index))

def find_evalharness(x):
    x=str(x).strip()
    for reg in [",", r"\\$", r"(?s).*#### ", r"\.(\*\*)?$"]:
        x = re.sub(reg, "", x)
    #x = x.split("The answer is ")[-1]
    f = re.findall(r"(-?[0-9\.\,]+)", x) # "#### (\\-?[0-9\\.\\,]+)"
    f = [i for i in f if i !="." and i !=","]
    if len(f)>0:
        try:
            return f[-1] #float(f[-1])
        except ValueError:
            return None
    return None

testset["hops"] = testset["question"].apply(lambda x: len(re.findall(r"\d+.*? ",x))-1)
testset["calc"] = testset["answer"].apply(lambda x: x.count(">>"))
testset["answer"] = testset["answer"].apply(lambda x: x.split("#### ")[-1].replace(",", "").strip())
model_rename = {'LLaDA-1.5':'LLaDA-1.5', 'Dream-v0-Instruct-7B':"Dream-v0", 'LLaDA2.0-mini':"LLaDA2.0-mini", 'Qwen3-8B':'Qwen3', 'Llama-3.1-8B-Instruct': 'Llama-3.1'}

accs_tables = pd.DataFrame(columns=["model", "numeos", "eval", "hops"])#, "calc"])

for file in os.scandir(f"./outfiles/exp4"):
    if "eval" in file.name or not "gsm8k" in file.name:
        continue
    results = pd.read_csv(file, sep="\t", names=["prediction", "prompt"])
    print("Length ", len(results.index))

    results["target"] = testset["answer"]
    results["hops"] = testset["hops"]
    results["calc"] = testset["calc"]
    results["model"] = model_rename.get(file.name.split("_")[1], file.name.split("_")[1])
    results["shots"] = file.name.split("shot")[0].split("_")[-1]
    results["setup"] = folder.replace("outfiles_gsm8k_","")
    results["numeos"] = int(file.name.split("numeos")[1].split("_")[0]) if "numeos" in file.name else 200

    results["number"] = results["prediction"].apply(find_evalharness)
    results["eval"] = results.apply(lambda x: x["target"] == x["number"], axis=1)
    
    print(f"File: {file.name} - Accuracy: {results['eval'].mean()}")
    results.to_csv(file.path + "eval.tsv", sep="\t", index=False)
    accs_tables = pd.concat([accs_tables, results[["model", "numeos", "eval", "hops", "setup", "shots"]]], axis=0)

accs_tables.to_csv("./eacl_deadline/gsm8k_all_accs.tsv", sep="\t", index=False)


g = sns.relplot(accs_tables, x="numeos", y="eval", kind="line", hue="model", errorbar="se",
hue_order=['Dream-v0', 'LLaDA-1.5', 'LLaDA2.0-mini'], 
palette={'LLaDA-1.5':"tab:green", 'Dream-v0':"tab:blue", 'LLaDA2.0-mini':"tab:purple"},)

plt.ylim(0,0.55)
for ax in g.axes.flat:
    ax.set_xscale("log", base=2)
g.set(xticks=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], xticklabels=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])#, base=2)
g._legend.set_title("Models")
plt.title("GSM8K")
plt.ylabel("Accuracy")
plt.xlabel("EoS Tokens")
plt.xticks(fontsize=10)
plt.savefig(f"./gsmk8_accs_per_eos.pdf", format="pdf")

print("Done")
