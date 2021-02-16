


bleurt_checkpoint=/mnt/scratch/youngwookim/code/bleurt/bleurt-base-128

for data_name in clue tdlt; do 
    for method in query random; do
        pred_path=output/genex/runs/${data_name}_${method}.txt
        for gold_idx in {0..11}; do
            gold_file_path=data/genex/${data_name}_gold/${gold_idx}
            if test -f "$gold_file_path"; then
                score_path=output/genex/scores/${data_name}_${method}_${gold_idx}.txt
                python -m bleurt.score \
                       -candidate_file=$pred_path \
                       -reference_file=$gold_file_path \
                       -bleurt_checkpoint=$bleurt_checkpoint \
                       -scores_file=$score_path 
                exit
            fi
        done
    done
done

