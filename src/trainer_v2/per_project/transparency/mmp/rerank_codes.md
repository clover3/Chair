
common_src_root = src/trainer_v2/per_project/transparency/mmp

## Rerank

### BM25 like

* BM25 rerank on mmp_dev
  * Hard coded w/ NLTKKrovetz: ${common}/runner/bm25_rerank.py
  * With config: ${common}/bm25_runner/run_bm25_rerank.py
* BM25T
  * ${common}/bm25t_runner/run_rerank_w_conf.py
* BM25T2
  * ${common}/pep/bm25t2/run_rerank.py

### PEP

* Architecture aware
  * From pairwise TS_add: ${common}/pep/runner/run_rerank_ts_score_add.py
* Architecture agnostic
  * PEP Scorer (Two segment based): ${common}/pep/pep_rerank.py


### Common Library
* ${common}/eval_helper/rerank_w_conf.py




## Full Retrieval

* ${common}/retrieval_run/run_bm25t.py