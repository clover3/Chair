@echo off
set label_path=C:\work\Code\Chair\output\alamri_annotation1\label\sbl.qrel.val
set run_name=slr_mismatch
set prediction_path=C:\work\Code\Chair\output\alamri_annotation1\ranked_list\%run_name%.txt
C:\work\Tool\trec_eval\trec_eval.exe -m map %label_path% %prediction_path%
C:\work\Tool\trec_eval\trec_eval.exe -m num_q %label_path% %prediction_path%