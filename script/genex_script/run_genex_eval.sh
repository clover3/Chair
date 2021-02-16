


bleurt_checkpoint=/mnt/scratch/youngwookim/code/bleurt/bleurt-base-128

for data_name in clue tdlt; do 
    for method in lime deletion; do
        for config in 1a 1b 2a; do
            pred_path=output/genex/runs/${data_name}_${method}_${config}.txt
            for gold_idx in {0..11}; do
                gold_file_path=data/genex/${data_name}_gold/${gold_idx}
                if test -f "$gold_file_path"; then
                    score_path=output/genex/scores/${data_name}_${method}_${config}_${gold_idx}.txt
                    python -m bleurt.score \
                           -candidate_file=$pred_path \
                           -reference_file=$gold_file_path \
                           -bleurt_checkpoint=$bleurt_checkpoint \
                           -scores_file=$score_path 
                fi
            done
        done
    done
done

