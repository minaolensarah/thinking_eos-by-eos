import pandas as pd 
import argparse
import re
import seaborn as sns
import os
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import argparse

sns.set_style("whitegrid")



def eval_maths(results, testset, cot):
    results["result"] = testset["result"]
    if cot:
        #results["cot"] = results["prediction"].apply(lambda x: x.split("final result is")[0])
        results["prediction"] = results["prediction"].apply(lambda x: x.lower().split("final result is")[-1])
    results["number"] = results["prediction"].apply(lambda x: int((re.findall(r"-?\d+", x) + [100000])[0]) if isinstance(x, str) else x)
    results["eval"] = results.apply(lambda x: int(x["result"]) == x["number"], axis=1)
    results["problem_length"] = testset["num_sumands"]
    #results["length"] = results["cot"].apply(lambda x: len(tokenizer(str(x))["input_ids"]))
    return results

objects = list(pd.read_csv("../datasets/objects_with_bnc_frequency.csv", sep=",")["object_name"])
x = {o: len(o) for o in objects}
objects = [k for k, _ in sorted(x.items(), key=lambda item: item[1], reverse=True)]
objects_pattern = "|".join(objects) 

def get_objs(text, doubleIsCorrect=False, nothing=True):
    if isinstance(text, str):
        text = text.lower()
    else:
        print("Text is not str:", text)
        text = ""
    objs_pred = list(re.findall(objects_pattern, text))
    if ("nothing" in text or "empty" in text) and nothing:
        objs_pred.append("nothing")
    return sorted(list(set(objs_pred))) if doubleIsCorrect else sorted(objs_pred)

def clean(text):
    text = text.lower().split("box")[1]
    if "." in text:
        text = text.split(".")[0]
    return text

def eval_box(results, testset, cot):
    results["problem_length"] = testset["numops"]
    if cot:
        results["prediction"] = results["prediction"].apply(lambda x: x.lower().split("box")[-1])

    results["obj_target"] = testset.apply(lambda x: get_objs(x["masked_content"]), axis=1)
    results["obj_pred"] = results.apply(lambda x: get_objs(x["prediction"]), axis=1)

    results["eval"] = results["obj_target"] == results["obj_pred"]
    return results

def validate_sudoku(sudoku):
    #print(sudoku)
    sudoku = sudoku.replace("\n", "").strip()
    sudoku = [int(i) for i in sudoku]
    nums = {1,2,3,4}

    # validate rows
    for i in range(4, 17, 4):
        if set(sudoku[(i-4):i]) != nums:
            return False
    # validate columns
    mod = [[], [], [], []]
    for ix, i in enumerate(sudoku):
        mod[ix%4].append(i)
    for i in mod:
        if set(i) != nums:
            return False
    #validate cells
    for i in [0, 2, 8, 10]:
        if {sudoku[i], sudoku[i+1], sudoku[i+4], sudoku[i+5]} != nums:
            return False 
    return True

def eval_sudoku(results, testset, cot):
    results["problem_length"] = testset["empty_cells"]
    
    results["clean"] = results["prediction"].apply(lambda x: "".join(re.findall(r"\d", x)[-16:]))
    results["eval"] = results["clean"].apply(validate_sudoku)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evalution script")
    parser.add_argument("--infolder", type=str, required=True, help="Path to the input data folder")
    parser.add_argument("--keyword", type=str, default=".", help="string that must be in the file name")
    parser.add_argument("--outfolder", type=str, help="Path to the output data folder")


    args = parser.parse_args()

    maths_testset = pd.read_json("../datasets/easy_maths_200perCalc.jsonl", lines=True, orient="records")
    box_testset = pd.read_json("../datasets/boxes_testset_24X30_pair1.jsonl", lines=True, orient="records")
    sudoku_testset = pd.read_json("../datasets/sudoku4x4_200_per_empty_cell1to12.jsonl", lines=True, orient="records")

    big_table = pd.DataFrame(columns=["filename", "model", "param", "task", "accuracy", "datasetlength"])

    for file in os.scandir(args.infolder):
        if args.keyword in file.name and not "eval" in file.name:
            cot = "cot" in file.name 
            results = pd.read_csv(file, sep="\t", names=["prediction", "prompt"])

            if "qwen" in file.name.lower() and not cot:
                results["prediction"] = results["prediction"].apply(lambda x: x.split("</think>")[-1])
            elif "llama" in file.name.lower() or "qwen" in file.name.lower():
                results["prediction"] = results["prediction"].apply(lambda x: x.split("assistant")[-1])

            if "maths" in file.name:
                task = "maths"
                results = eval_maths(results, maths_testset, cot)
            elif "box" in file.name:
                task = "boxes"
                results = eval_box(results, box_testset, cot)
            elif "sudoku" in file.name:
                task = "sudoku"
                results = eval_sudoku(results, sudoku_testset, cot)

            print(f"File: {file.name} - Accuracy: {results['eval'].mean()} - Length: {len(results.index)}")
            
            if "genlength" in file.name:
                param = int(file.name.split("genlength")[1].split("_")[0])
            elif "eos" in file.name:
                param = int(file.name.split("eos")[1].split("_")[0])
            else:
                param = None

            big_table.loc[(len(big_table.index))] = [file.name, file.name.split("_")[0], param, task, results['eval'].mean(), len(results.drop_duplicates(subset=["prompt"]).index)]

            if args.outfolder is not None:
                results[["problem_length", "eval"]].to_csv(os.path.join(args.outfolder, f"{file.name}_eval.csv"), sep="\t", index=False)

    big_table.to_csv(os.path.join(args.outfolder, f"accuracies.csv"), sep="\t", index=False)

if __name__ == "__main__":
    main()