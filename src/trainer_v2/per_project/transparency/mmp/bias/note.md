
### Step 1. Make corpus

save_car_queries.py
save_car_maker_document.py
car_queries_filter_makers.py


## Step 2. First step inference to filter query/doc

run_qd_pair_inference.py

* Select high scored query/document

## Step 3. Swap Car maker and score it

run_inference_w_keyword_swap.py
run_inf_w_keyword_swap_dev_split.py
* 5152 docs (matched_doc_dev_all.tsv)
* 2254 queries (dev_matched_car_queries_no_maker.tsv)


## Step 4. Run Analysis
  
* If the original document contains a car maker A, 
  * We assume it is more likely to relevant to A even after the replacing the car maker
  * We assume it should NOT show bias to specific car maker B over car maker C.
* 