

## pipeline 1

* Relevance scoring
  * perspective_paragraph_feature
  * run_build_pc_rel_payload_dev.pickle
  * pc_rel_info = "pc_rel_dev_info_all"
  * pc_rel : NN prediction
  * pc_rel_based_score
  
## Pipeline2

* Relevance scoring
  * perspective_paragraph_feature
  * run_build_pc_rel_payload_dev.pickle
  * pc_rel_dev_with_cpid.pickle 
* Classification
  * pc_rel_filter_dev.py
  * 