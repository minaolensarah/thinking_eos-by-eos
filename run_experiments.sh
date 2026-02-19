
# run the first experiment with Dream v0
for genlength in 20 24 32 40 48 56 64 72 80
do
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --sudoku --batch_size 48 --gen_length $genlength  --outfolder "outfiles/exp1/" --data_path "./datasets/sudoku4x4_200_per_empty_cell1to12.jsonl"
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --maths --batch_size 48 --gen_length $genlength  --outfolder "outfiles/exp1/" --data_path "./datasets/easy_maths_200perCalc.jsonl"
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --boxes --batch_size 48 --gen_length $genlength  --outfolder "outfiles/exp1/" --data_path "./datasets/boxes_testset_24X30_pair1.jsonl"
done

python3 eval_AddBoxSud.py --infolder "outfiles/exp1/" --outfolder "outfiles/exp1_eval/" 

# run the second experiment with Dream v0
for numeos in 1 2 4 8 16 24 32 64 128
do
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --sudoku --batch_size 48  --num_eos $numeos --num_masks 19 --steps 19 --outfolder "outfiles/exp2/" --data_path "./datasets/sudoku4x4_200_per_empty_cell1to12.jsonl"
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --maths --batch_size 48  --num_eos $numeos --num_masks 12 --steps 12 --outfolder "outfiles/exp2/" --data_path "./datasets/easy_maths_200perCalc.jsonl"
    python3 src/prompting_experiments.py --model "Dream-org/Dream-v0-Instruct-7B" --boxes --batch_size 48 --num_eos $numeos --num_masks 22 --steps 22  --outfolder "outfiles/exp2/" --data_path "./datasets/boxes_testset_24X30_pair1.jsonl"
done

python3 eval_AddBoxSud.py --infolder "outfiles/exp2/" --outfolder "outfiles/exp2_eval/" 


# run the activation patching with LLaDA1.5 for Sudoku
mkdir activation_patching_results
python3 src/activation_patching.py --task "sudoku" --model "Dream-org/Dream-v0-Instruct-7B"
mkdir patching_tables
python3 src/evaluate_patching_sudoku.py --infolder "activation_patching_results"

# run the CoT comparison
python3 src/prompting_experiments.py --boxes --cot --model "Dream-org/Dream-v0-Instruct-7B" --batch_size 32 --gen_length 1504 --block_length 32 --steps 752 --outfolder "outfiles/exp4/" --data_path "./datasets/boxes_testset_24X30_pair1.jsonl"
python3 src/prompting_experiments.py --sudoku --cot --model "Dream-org/Dream-v0-Instruct-7B" --batch_size 32 --gen_length 1504 --block_length 32 --steps 752 --outfolder "outfiles/exp4/" --data_path "./datasets/sudoku4x4_200_per_empty_cell1to12.jsonl"
python3 src/prompting_experiments.py --maths --cot --model "Dream-org/Dream-v0-Instruct-7B" --batch_size 32 --gen_length 1504 --block_length 32 --steps 752 --outfolder "outfiles/exp4/" --data_path "./datasets/easy_maths_200perCalc.jsonl"

python3 eval_AddBoxSud.py --infolder "outfiles/exp4/" --outfolder "outfiles/exp4_eval/"

