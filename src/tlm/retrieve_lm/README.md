
1. sample_segment.py
   * robust_segments_
      * target_tokens : subword 
      * sent_list
      * prev_tokens : subword
      * next_tokens : subword
1. segment2problem.py
   * robust_segment &rarr; robust_problem
2. retrieve_candidates.py
   * robust_problem &rarr; robust_problem_q, robust_candi_query
3. segment_ranker_0.py
   * robust_problem_q_ &rarr; CandiSet
   * CandiSet : (target,mask_indice,prev,next, passages,sent_list,doc_id)
      * Target, mask_indice : Problem with mask
      * prev, next
      * passage : doc_id, loc
4. tf_record_writer.py : 
   * CandiSet &rarr; TFRecord, History
   
   * predict.        
    