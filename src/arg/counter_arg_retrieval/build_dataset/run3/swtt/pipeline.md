
root: src/arg/counter_arg_retrieval/build_dataset/run3/

7 queries.

docs.jsonl

* doc_processing.py: Process jsonl
  *  output: ca_run3_swtt.pickle
    
* run_cls_jobs.py
  *  output: .result 
    
* swtt/save_trec_style.py
  *  output: {}_ranked_list\{run_name}

* slice_ranked_list.py
* print_to_csv.py