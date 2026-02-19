import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import editdistance
import re
import argparse

sns.set_style("whitegrid")

def is_the_same_as(inp, clean, corr, patched):
    inp = inp.replace("\n", "")
    clean = "".join(re.findall(r"\d", clean)[-16:])
    corr = "".join(re.findall(r"\d", corr)[-16:])
    patched = "".join(re.findall(r"\d", patched)[-16:])
    return editdistance.eval(patched, clean)/16, editdistance.eval(patched, corr)/16

def main(infolder, keyword):
    pairs1 = pd.read_json("../datasets/sudoku4x4_200_per_empty_cell1to12.jsonl", orient='records', lines=True)
    pairs1["converted_puzzle"] = pairs1["converted_puzzle"].apply(lambda x: (16-len(str(x)))*"0"+str(x))
    mapping = ["0", "2", "3", "4", "1"]
    globaldf = pd.DataFrame(columns = [ "eos", "layer", "distance", "to"])
    columns=["sentence_id", "layer", "clean_input", "corrupted_input", "clean_output", "corrupted_output", "patched_output"]

    for file in os.scandir(infolder):
        if "eval" not in file.name and keyword in file.name:
            print(file.name)
            data = pd.read_csv(file, sep="\t", names=columns)
            eos = int(file.name.split("eos")[1].split(".")[0])
            data["eos"] = len(data) * [eos]
            data["input_sudoku"] = pairs1["converted_puzzle"]
            data["match"] = data.apply(lambda x: is_the_same_as(x["clean_input"], x["clean_output"], x["corrupted_output"], x["patched_output"]), axis=1)
            data["match_clean"], data["match_corrupted"] = zip(*data["match"])

            data["distance"] =data["match_clean"] 
            data["to"] = len(data)* ["clean"]
            globaldf = pd.concat([globaldf, data[[ "eos", "layer", "distance", "to"]]], ignore_index=True)
            data["distance"] =data["match_corrupted"] 
            data["to"] = len(data)* ["corrupted"]
            globaldf = pd.concat([globaldf, data[[ "eos", "layer", "distance", "to"]]], ignore_index=True)

    globaldf[globaldf["layer"]!="layer"].to_csv(f"../patching_tables/{args.keyword}sudoku_activation_patching_overall_eval.tsv", sep="\t", index=False)

    eoses = set(globaldf["eos"])


    for eos in eoses:   
        sns.relplot(globaldf[globaldf["eos"]==eos], x="layer", y="distance", hue="to", kind="line",errorbar=None)
        plt.legend(loc='upper left')
        plt.title(f"EOS {eos}: Distance from the patched output sudoku")
        plt.xlabel("Layer")
        plt.ylabel("Normalized Levenshtein Distance")
        plt.ylim(0,1)
        plt.savefig(f"./plots_patching/line_patching_sudoku_fullmasked_eos{eos}.png")
        plt.clf()

    print("Done evaluating sudoku")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Evalution script")
    parser.add_argument("--infolder", type=str, required=True, help="Path to the input data folder")
    parser.add_argument("--keyword", type=str, default="sudoku", help="string that must be in the file name")

    args = parser.parse_args()
    main(args.infolder, args.keyword)