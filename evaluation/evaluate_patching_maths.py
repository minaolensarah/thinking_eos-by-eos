import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import argparse

sns.set_style("whitegrid")

def get_flipped__result(calculation):
    flipped = ""
    for letter in calculation:
        if letter =="+":
            letter = "-"
        elif letter =="-":
            letter = "+"
        flipped += letter
    return eval(flipped)

def is_the_same_as(flipped, og, output):
    output = output.split("result")[-1]
    result = re.findall(r"-?\d+", output)
    if len(result) > 0:
        result = int(result[-1])
    else:
        result = None
    returnvalue = "neither"
    if result == int(og):
        returnvalue = "clean"
    elif flipped == result:
        returnvalue = "corrupted"
    return returnvalue

def main(infolder, keyword):
    pairs1 = pd.read_json("../datasets/easy_maths_200perCalc.jsonl", orient='records', lines=True)
    pairs1["flipped_result"] = pairs1["calculation"].apply(get_flipped__result)

    globaldf = pd.DataFrame(columns = [ "eos", "layer", "match"])
    columns=["sentence_id", "layer", "clean_input", "corrupted_input", "clean_output", "corrupted_output", "patched_output"]

    for file in os.scandir(infolder):
        #print(file.name)
        if "eval" not in file.name and keyword in file.name:
            print(file.name)
            data = pd.read_csv(file, sep="\t", names=columns)
            eos = int(file.name.split("eos")[1].split(".")[0])
            data["eos"] = len(data) * [eos]
            #layer = int(file.name.split("layer")[1].split("_")[0])
            #data["og_result"] = pairs1["result"]
            data["match"] = data.apply(lambda x: is_the_same_as(pairs1.iloc[x["sentence_id"]]["flipped_result"], pairs1.iloc[x["sentence_id"]]["result"], x["patched_output"]), axis=1)
            
            globaldf = pd.concat([globaldf, data[[ "eos", "layer", "match"]]], ignore_index=True)

    globaldf.to_csv(f"../patching_tables/{args.keyword}maths_activation_patching_overall_eval.tsv", sep="\t", index=False)



    eoses = set(globaldf["eos"])


    for eos in eoses:   
        data["clean"] = data["match"].apply(lambda x: x =="clean")
        data["corrupted"] = data["match"].apply(lambda x: x=="corrupted")
        g = data[["clean", "corrupted","layer"]].groupby(["layer"]).mean()

        plt.stackplot(sorted(list(set(data["layer"]))),g["clean"], g["corrupted"], labels=["clean", "corrupted"])
        plt.legend(loc='upper left')
        plt.title(f"EOS {eos} perfect match")
        plt.xlabel("Layer")
        plt.ylim(0,1)
        plt.savefig(f"../plots_patching/patching_maths_distributions_PERFECT_MATCH_eos{eos}.png")
        plt.clf()

    print("Done evaluating maths")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalution script")
    parser.add_argument("--infolder", type=str, required=True, help="Path to the input data folder")
    parser.add_argument("--keyword", type=str, default="maths", help="string that must be in the file name")

    args = parser.parse_args()
    main(args.infolder, args.keyword)
