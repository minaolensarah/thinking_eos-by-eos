import pandas as pd 
from collections import Counter
import seaborn as sns 
import os
import matplotlib.pyplot as plt
import re

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=2)


cols = ["sentence_id", "layer", "clean_input", "corrupted_input", "clean_output", "corrupted_output", "patched_output", "Counterfactual Tokens in Original", "Counterfactual Tokens in Patched", "Orignal Tokens in Patched"]
all_data = pd.DataFrame()
all_reformat = pd.DataFrame()
counter = 0
for file in os.scandir("activation_patching_results"):
    if "64" in file.name:
        data = pd.read_csv(file, sep="\t", names=cols)
        print(file.name)
        print("Total length", len(data.index))
        # if the counterfactual generation is the same as the original generation skip datapoint from eval
        data = data[data["Counterfactual Tokens in Original"]!=1]
        if "maths" in file.name or "boxes" in file.name:
            if "random" in file.name and "maths" in file.name:
                def count_digits(x):
                    x = x.split("?")[-1]
                    return len(re.findall(r"-?\d+",x))
                data["corrupted_only_contains_filler"] = data["corrupted_output"].apply(lambda x: count_digits(x)==0)
                #remove one outlier where filler starts with <|mask|>１ doğ_journal耨因为在 нагруз缊رياض�Dange leading to extremly high ranks
                data = data[data["sentence_id"]!=770]
            else:
                split_at = "?" if "maths" in file.name else "contains"
                def remove_filler(x):
                    x = x.split(split_at)[-1]
                    return x.replace(" ", "").replace(".", ""). replace("<|endoftext|>", "").replace("<|eot_id|>", "")
                data["corrupted_only_contains_filler"] = data["corrupted_output"].apply(lambda x: len(remove_filler(x))==0)
            data = data[data["corrupted_only_contains_filler"]==False]
            
        print("valid length", len(data.index))

        if "Dream" in file.name:
            data["nummasks"] = data["clean_input"].apply(lambda x: x.count("<|mask|>"))
        else:
            data["nummasks"] = data["clean_input"].apply(lambda x: x.count("<|mdm_mask|>"))

        if "eos_" in file.name or "eos.csv" in file.name:
            patch_with = "EoS"
        elif "whitespace" in file.name:
            patch_with = "Whitespaces"
        elif "dots_" in file.name:
            patch_with = "Dots"
        elif "seed67" in file.name:
            patch_with = "Random 1"
        else:
            patch_with = "Random 2"

        reformat = pd.DataFrame()
        thislist = []
        thislist.extend(data["Counterfactual Tokens in Original"]) 
        thislist.extend(data["Counterfactual Tokens in Patched"]) 
        thislist.extend(data["Orignal Tokens in Patched"]) 
        
        reformat["avg"] = thislist
        reformat["layer"] = 3 * list(data["layer"])
        reformat["rank_metric"] = len(data.index) * ["Counterfactual Tokens in Original"] + len(data.index) * ["Counterfactual Tokens in Patched"] +len(data.index) * ["Orignal Tokens in Patched"]
        reformat["model"] = ("Dream-v0") if "Dream" in file.name else "LLaDA-1.5"
        reformat["task"] = ("Entity Tracking") if "boxes" in file.name else ("Sudoku" if "sudoku" in file.name else "Addition")
        reformat["numeos"] = int(file.name.split("eos")[1][:2])
        reformat["patch_with"] = patch_with

        all_reformat = pd.concat([all_reformat, reformat])


for patch_with in all_reformat["patch_with"].unique():
    g = sns.relplot(all_reformat[(all_reformat["numeos"]==64)&(all_reformat["patch_with"]==patch_with)], x="layer", y="avg", 
    hue="rank_metric", kind="line", col="task", row="model",
    col_order=["Addition", "Entity Tracking", "Sudoku"],  
    row_order=['Dream-v0', 'LLaDA-1.5'],
    errorbar="se",facet_kws={ "sharey":"row","sharex":False}) #

    g.set(xlabel="Layer", ylabel="Rank")
    g._legend.set_title("Average Rank (with standard error) of")
    sns.move_legend(g, loc="lower center", bbox_to_anchor=(0.4, -0.1), ncols=3)

    ax_right = g.axes[0][2].twinx()
    ax_right.set_yticklabels([])
    ax_right.set_yticks([])
    ax_right.set_ylabel("Dream-v0", rotation=270, labelpad=30)

    ax_right = g.axes[1][2].twinx()
    ax_right.set_yticklabels([])
    ax_right.set_yticks([])
    ax_right.set_ylabel("LLaDA-1.5", rotation=270, labelpad=30)

    g.axes[0][0].set_title("Addition")
    g.axes[0][1].set_title("Entity Tracking")
    g.axes[0][2].set_title("Sudoku")
    g.axes[1][0].set_title("")
    g.axes[1][1].set_title("")
    g.axes[1][2].set_title("")

    for i in range(1,3):
        g.axes[0][i].set_ylabel("")
        g.axes[1][i].set_ylabel("")

    g.savefig(f"absolute_ranks_all_{patch_with}.pdf", format='pdf')
    plt.close()

print("Done with per filler")

for task in ["Addition", "Entity Tracking", "Sudoku"]:
    g = sns.relplot(all_reformat[(all_reformat["numeos"]==64)&(all_reformat["task"]==task)], x="layer", y="avg", 
    hue="rank_metric", kind="line", col="patch_with", row="model",
    col_order=["EoS", "Dots", "Whitespaces", "Random 1", "Random 2"],  
    row_order=['Dream-v0', 'LLaDA-1.5'],
    errorbar="se",facet_kws={ "sharey":"row","sharex":False}) 

    g.set(xlabel="Layer", ylabel="Rank")
    g._legend.set_title("Average Rank (with standard error) of")
    sns.move_legend(g, loc="lower center", bbox_to_anchor=(0.4, -0.1), ncols=3)

    ax_right = g.axes[0][4].twinx()
    ax_right.set_yticklabels([])
    ax_right.set_yticks([])
    ax_right.set_ylabel("Dream-v0", rotation=270, labelpad=30)

    ax_right = g.axes[1][4].twinx()
    ax_right.set_yticklabels([])
    ax_right.set_yticks([])
    ax_right.set_ylabel("LLaDA-1.5", rotation=270, labelpad=30)
    
    g.axes[0][0].set_title("EoS")
    g.axes[0][1].set_title("Dots")
    g.axes[0][2].set_title("Whitespace")
    g.axes[0][3].set_title("Random 1")
    g.axes[0][4].set_title("Random 2")

    g.axes[1][0].set_title("")
    g.axes[1][1].set_title("")
    g.axes[1][2].set_title("")
    g.axes[1][3].set_title("")
    g.axes[1][4].set_title("")

    for i in range(1,5):
        g.axes[0][i].set_ylabel("")
        g.axes[1][i].set_ylabel("")

    g.savefig(f"absolute_ranks_all_{task}.pdf", format='pdf')
    plt.close()

print("Done with per task")

