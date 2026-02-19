import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set_style("whitegrid")

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

def split_at_last_box(text):
    return text.lower().split("box")[-1]

def main(infolder, keyword):
    pairs1 = pd.read_json("../datasets/boxes_testset_24X30_pair1.jsonl", orient='records', lines=True)
    pairs2 = pd.read_json("../datasets/boxes_testset_24X30_pair2.jsonl", orient='records', lines=True)
    pairs1["target_objs"] = pairs1["masked_content"].apply(get_objs)
    pairs2["target_objs"] = pairs2["masked_content"].apply(get_objs)

    globaldf = pd.DataFrame(columns = ["sentence_id", "eos", "layer", "clean_correct", "corrupted_correct", "micro_avg_fliped","micro_avg_clean", "patched_is_clean", 
                "patched_is_corrputed",  "patched_is_neither", "patched_is_clean_correct", "patched_is_corrupted_correct", "patched_is_incorrect"])

    save_df = pd.DataFrame(columns=["eos", "layer", "macro_avg_clean", "macro_avg_fliped"])

    columns=["sentence_id", "layer", "clean_input", "corrupted_input", "clean_output", "corrupted_output", "patched_output"]

    for file in os.scandir(infolder):
        #print(file.name)
        if "eval" not in file.name and "boxes_allLayers" in file.name and keyword in file.name:
            print(file.name)
            eos = int(file.name.split("eos")[1].split(".")[0])
            #layer = int(file.name.split("layer")[1].split("_")[0])
            data = pd.read_csv(file, sep="\t", names=columns)
            for c in ["clean_output", "corrupted_output", "patched_output"]:
                data[c] = data[c].apply(split_at_last_box)
                data[c + "_objs"] = data[c].apply(get_objs)
            data["eos"] = eos
            results = pd.DataFrame(columns= list(data.columns) + [ "clean_correct", "corrupted_correct", "patched", "micro_avg_fliped", "micro_avg_clean"])
            for ix, row in data.iterrows():
                i = row["sentence_id"]
                clean_row = pairs1[pairs1["sentence_id"]==i].iloc[0]
                corrupted_row = pairs2[pairs2["sentence_id"]==i].iloc[0]

                # basic accuracies without patching
                clean_correct = row["clean_output_objs"] == clean_row["target_objs"]
                corrupted_correct = row["corrupted_output_objs"] == corrupted_row["target_objs"]

                # does patching result in a flip to the correct answer?
                if row["patched_output_objs"] == clean_row["target_objs"]:
                    patched = "clean"
                    correct_row = [1 ,0 ,0]
                elif row["patched_output_objs"] == corrupted_row["target_objs"]:
                    patched = "corrupted"
                    correct_row = [0, 1 ,0]
                else:
                    patched = "neither"
                    correct_row = [0 ,0, 1]
                
                # does patching result in a flip?
                if row["patched_output_objs"] == row["clean_output_objs"]:
                    patched_flip = "clean"
                    flip_row = [1 ,0 ,0]
                elif row["patched_output_objs"] == row["corrupted_output_objs"]:
                    patched_flip = "corrupted"
                    flip_row = [0, 1 ,0]
                else:
                    patched_flip = "neither"
                    flip_row = [0 ,0, 1]
                
                micro_avg_fliped = 0
                micro_avg_clean = 0
                if len(row["patched_output_objs"]) > 0:
                    for obj in row["patched_output_objs"]:
                        if (obj in row["clean_output_objs"] or obj in clean_row["target_objs"]):
                            micro_avg_clean+=1
                        elif (obj in row["corrupted_output_objs"] or obj in corrupted_row["target_objs"]):
                            micro_avg_fliped+=1
                    micro_avg_fliped /= len(row["patched_output_objs"])
                    micro_avg_clean /= len(row["patched_output_objs"])


                new = [clean_correct,
                    corrupted_correct,
                    patched, micro_avg_fliped, micro_avg_clean]
                    
                results.loc[len(results)] = list(row) + new
                globaldf.loc[len(globaldf)] = [i, eos, row["layer"], clean_correct, corrupted_correct, micro_avg_fliped, micro_avg_clean] + correct_row + flip_row
                
            #results.to_csv(f"../activation_patching_results/{file.name}_eval.tsv", sep="\t", index=False)
            print("Clean accuracy: ", results["clean_correct"].mean())
            print("Corrupted accuracy: ", results["corrupted_correct"].mean())
            print("Patched results: ")
            print(results["patched"].value_counts(normalize=True))
    globaldf.to_csv(f"../patching_eval/{args.keyword}activation_patching_overall_eval.tsv", sep="\t", index=False)


    eoses = set(globaldf["eos"])



    for eos in eoses:   
        g = globaldf[globaldf["eos"]==eos].groupby(["layer"]).mean()
        save_df = pd.DataFrame()
        save_df["layer"] = g.index
        save_df["macro_avg_clean"] = list(g["micro_avg_clean"])
        save_df["macro_avg_fliped"] = list(g["micro_avg_fliped"])
        save_df["eos"] = len(save_df) *[eos]
        save_df.to_csv(f"../outfiles_paper/patching_exp/patching_tables/{args.keyword}_eos{eos}_boxes_patching.csv", sep="\t", index=False)

    print("Done evaluating boxes")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evalution script")
    parser.add_argument("--infolder", type=str, required=True, help="Path to the input data folder")
    parser.add_argument("--keyword", type=str, default=".", help="string that must be in the file name")


    args = parser.parse_args()
   main(args.infolder, args.keyword)