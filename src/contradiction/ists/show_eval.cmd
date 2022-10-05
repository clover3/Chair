@echo off
set label_path=C:\work\Code\Chair\output\ists\noali_label\headlines_train.qrel
set run_name=nlits_punc_nc
set prediction_path=C:\work\Code\Chair\output\ists\noali_pred\%run_name%.txt
C:\work\Tool\trec_eval\trec_eval.exe -m map %label_path% %prediction_path%
C:\work\Tool\trec_eval\trec_eval.exe -m num_q %label_path% %prediction_path%