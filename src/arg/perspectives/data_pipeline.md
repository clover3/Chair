

* Claim+Perspective  --(query_gen.py) --> Query
  * dir: Chair/output/perspective_train_claim_perspective_query
* Query --(run_pc_query.py) --> Disk #i -> Ranked_list
  * Ranked_list : work3_data/perspective/train_claim_perspective/all_query_results/
    * perspective_train_claim_perspective_query/10_diskname.txt
* Ranked_list --(run_pc_jsonl_jobs.py)--> DB
