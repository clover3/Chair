

1. merge_paragraph_claim_feature.py -> "pc_dev_paras_by_cid"
2. build_graph_from_token.py -> "pc_dev_co_occur"
3. run_pc_dev_random_walk.py

4. merge_pickled_results.py 


---

Get Pos/Neg from train claim/pers

1. RandomWalk(Neg, Passages) -> Get neg indicators


---

1. merge_paragraph_claim_feature.py -> "pc_dev_paras_by_cid"

"pc_dev_paras_top_100" / pc_train_paas_by_cid.pickle

2. unpack_score_paragraph.py

  * "pc_dev_paras_top_100_list_form"

3. train_word2vec.py 

  * sydney_working/pc_train_word2vec
  * (key: str, model) 