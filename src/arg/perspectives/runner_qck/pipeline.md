

1. run_qk_candidate_gen.py : First candidate
 (Query, Knowledge document (all) )

1. qcknc_datagen.py : To predict < train split2 > validation set
 (Query, Knowledge document (all) )

1. FinalScorer Predict < train split2 >
    To select which QK is good 

1. Predict passages on < train split >
  * run_regression
  * output : qknr2_train

1. doc_scorer_summarizer.py  < train split >
  * Select good passages 
  * output: perspective_qk_candidate2_train.pickle

3. qcknc_datagen2.py : To train second FinalScorer < train split >
 
 
3. run_qk_candidate_gen.py : Make all QK candidates for < dev split >
 perspective_qk_candidate_val
 
4. qk_regression_datagen_dev.py : < dev split >
    Make data to be fed to doc selector 

1. doc_scorer_summarizer.py  < dev split >
  * Select good passages 
  * output: perspective_qk_stage2_dev.pickle
 
4. qcknc_pred_datagen.py : payload for dev set
