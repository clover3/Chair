

Input: Ranked List, Jsonl

1. Remove duplicate documents, make new ranked list
   Output: ranked list
   * src/arg/counter_arg_retrieval/build_dataset/run3/run_interface/duplicate_removal.py
2. Split each document into passages
    Old implementation: DocumentScorerSWTT

    SWTTScorerInput
    
    Doc = [SWTT, List[SWTTScorerInput]]
    Output: Dict[doc_id, Doc]
    * src/arg/counter_arg_retrieval/build_dataset/run3/run_interface/split_documents.py
    
3. Load ranked list and apply scorer to each passage.
    




