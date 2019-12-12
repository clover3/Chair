
# Procedure for TLM Training
* We have LM Models
1. tlm/tlm/run_predict.py : predict LM loss ( from either bfn or bert)
1. tlm/tlm/gen_bfn_loss_predict.py : Combine two loss files and making TFRecord
1. tlm/training/runner/train_loss_predict.py : Train model that predict loss of bert and bfn
    1. We now have LossPredicter. 
1. tlm/training/dynamic_mask_main.py : Train targeted LM
    1. tlm2_train.sh 
