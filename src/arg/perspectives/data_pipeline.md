

* Claim+Perspective  --(query_gen.py) --> Query
  * dir: Chair/output/perspective_train_claim_perspective_query
* Query --(run_pc_query.py) --> Disk #i -> Ranked_list
  * slurm jobs are submitted
  * Ranked_list : work3_data/perspective/train_claim_perspective/all_query_results/
    * perspective_train_claim_perspective_query/10_diskname.txt
    
* Ranked_list --> fetch_doc_list.py --> doc_id_list
* doc_id_list --> galago get-doc-jsonl --> jsonl
* jsonl --(run_pc_jsonl_jobs.py)--> DB


* DB -> run_select_paragraph.py : -> features

*  