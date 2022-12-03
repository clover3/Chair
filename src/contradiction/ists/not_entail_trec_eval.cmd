@echo off
set label_path=C:\work\Code\Chair\output\ists\not_entail\headlines_train.qrel
set run_name=headlines_train_partial_seg
set prediction_path=C:\work\Code\Chair\output\ists\not_entail_pred\%run_name%.txt
C:\work\Tool\trec_eval\trec_eval.exe -m map %label_path% %prediction_path%
C:\work\Tool\trec_eval\trec_eval.exe -m num_q %label_path% %prediction_path%