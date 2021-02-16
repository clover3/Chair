export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=2
export LOGGER=4
model_path=/mnt/scratch/youngwookim/Chair/output/model/runs/BERT_Base_trained_on_MSMARCO/model.ckpt-100000


for name in wiki; do 
    for method in deletion LIME; do 
        echo $name
        python src/explain/genex/run_baseline.py \
            --model_path=$model_path \
            --data_name=${name} \
            --method_name=$method 
        
    done

done
