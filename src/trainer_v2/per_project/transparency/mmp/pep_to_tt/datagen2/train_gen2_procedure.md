


Corpus: get_train_triples_partition_path

${common}: src/trainer_v2/per_project/transparency/mmp/pep_to_tt/data_gen

1. ${common}/write_lookup_todo.py {conf} {job_no}
2. ${common}/table_lookup.py {conf} {worker_i}
3. ${common}/train_gen2.py {job_no}

