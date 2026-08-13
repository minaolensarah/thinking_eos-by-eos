
# run the first experiment with Dream v0
for genlength in 20 24 32 40 48 56 64 72 80
do
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --sudoku --batch_size 48 --gen_length $genlength  --outfolder "outfiles/exp1/" --data_path "./datasets/sudoku4x4_200_per_empty_cell1to12.jsonl"
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --maths --batch_size 48 --gen_length $genlength  --outfolder "outfiles/exp1/" --data_path "./datasets/easy_maths_200perCalc.jsonl"
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --boxes --batch_size 48 --gen_length $genlength  --outfolder "outfiles/exp1/" --data_path "./datasets/boxes_testset_24X30_pair1.jsonl"
done

python3 eval_AddBoxSud.py --infolder "outfiles/exp1/" --outfolder "outfiles/exp1_eval/" 

# run the second experiment with Dream v0

# for the additional experiment with other filler tokens use:
# dot: --eos_id 13
# whitespace: --eos_id 220
# random 1: --random_tokens --seed 42
# random 2: --random_tokens --seed 67

for numeos in 1 2 4 8 16 24 32 64 128
do
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --sudoku --batch_size 48  --num_eos $numeos --num_masks 19 --steps 19 --outfolder "outfiles/exp2/" --data_path "./datasets/sudoku4x4_200_per_empty_cell1to12.jsonl"
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --maths --batch_size 48  --num_eos $numeos --num_masks 12 --steps 12 --outfolder "outfiles/exp2/" --data_path "./datasets/easy_maths_200perCalc.jsonl"
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --boxes --batch_size 48 --num_eos $numeos --num_masks 22 --steps 22  --outfolder "outfiles/exp2/" --data_path "./datasets/boxes_testset_24X30_pair1.jsonl"
done

python3 src/evaluation/eval_AddBoxSud.py --infolder "outfiles/exp2/" --outfolder "outfiles/exp2_eval/" 


# run the activation patching with LLaDA1.5 for Sudoku
mkdir activation_patching_results
python3 src/activation_patching.py --task "sudoku" --model "Dream-org/Dream-v0-Instruct-7B" --pad_with "eos"
python3 src/activation_patching.py --task "sudoku" --model "Dream-org/Dream-v0-Instruct-7B" --pad_with "whitespace"

python3 src/plot_activation_patching.py 

# run the fourth experiment on more naturalistic datasets
python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --gsm8k --shots 5 --detailleddemo --temp01 --batch_size 4 --num_eos 128 --num_masks 18 --steps 18 --outfolder "outfiles_exp4/outfiles_gsm8k" --data_path "./datasets/test-canonical_gsm8k.jsonl"
python3 src/evaluation/eval_gsm8k.py

python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --twohop  --batch_size 80 --temp01 --num_eos 128 --num_masks 6 --steps 6 --outfolder "outfiles/exp4/" --data_path "datasets/two_hop_answerlenmax5.csv"
python3 src/evaluation/eval_2hop.py

