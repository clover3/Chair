

mct6 is from pep_tt16 100K predictions

It is using lucene-krovetz tokenizer. 

# Evalutions steps

./sync_table.sh
python src/trainer_v2/per_project/transparency/mmp/retrieval_run/bm25t_luk.py {table_name} {run_name}
