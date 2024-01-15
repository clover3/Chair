
common_src_root = src/trainer_v2/per_project/transparency/mmp

## Rerank

### BM25 like

* BM25 rerank on mmp_dev
  * Hard coded w/ NLTKKrovetz: ${common}/runner/bm25_rerank.py
  * With config: ${common}/bm25_runner/run_bm25_rerank.py
* BM25T
  * ${common}/bm25t_runner/run_rerank_w_conf.py
* BM25T2 (With PEP dynamic run)
  * ${common}/pep/bm25t2/run_rerank.py
* BM25T_3 (With max strategy)
  * ${common}/bm25t_runner/run_rerank_bm25t_3.py {conf}
  * conf: confs/experiment_confs/bm25t/bm25t_table5_dev1000.yaml

### PEP

* Architecture aware
  * From pairwise TS_add: ${common}/pep/runner/run_rerank_ts_score_add.py
* Architecture agnostic
  * PEP Scorer (Two segment based): ${common}/pep/pep_rerank.py


### Common Library
* ${common}/eval_helper/rerank_w_conf.py



## Full Retrieval

* ${common}/retrieval_run/run_bm25t.py {conf} 
* ${common}/retrieval_run/table_benchmark.py {table_path} {run_name}
  * run_retrieval_eval_report_w_conf

* src/trainer_v2/per_project/transparency/mmp/retrieval_run/run_bm25t_2.py {conf}
* 