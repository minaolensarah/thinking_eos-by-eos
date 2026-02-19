import pandas as pd 
import random

def generate_additions(min_calcs=2, max_calcs=7, num=200):
    df = pd.DataFrame(columns=["num_sumands", "calculation", "result"])
    if min_calcs < 2:
        min_calcs = 2
        print("Setting minimum calculations to 2")

    for _ in range(num):
        for calcs in range(min_calcs, max_calcs+1):
            valid = False
            while not valid:
                sumands = []
                for i in range(calcs):
                    sumands.append(random.randint(0,99))
                operators = []
                result = sumands[0]
                calculation = str(sumands[0])
                for s in range(1, calcs):
                    o = random.randint(0,1)
                    if o == 0:
                        operators.append("-")
                        result -= sumands[s]
                        calculation += f"-{sumands[s]}"
                    else:
                        operators.append("+")
                        result += sumands[s]
                        calculation += f"+{sumands[s]}"
                if result < 1000 and result > 0:
                    valid = True
            df.loc[len(df.index)] = [calcs, calculation, result] 
    return df

if __name__=="__main__":
    df = generate_additions()
    df.drop_duplicates(subset=["calculation"])
    df.to_json("./datasets/addition.jsonl", orient="records", lines=True)