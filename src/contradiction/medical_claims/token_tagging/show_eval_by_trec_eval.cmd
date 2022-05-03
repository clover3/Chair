@echo off
set label_path=C:\work\Code\Chair\output\alamri_annotation1\label\sbl.qrel.val
set run_name=exact_match_st_handle_mismatch
set prediction_path=C:\work\Code\Chair\output\alamri_annotation1\ranked_list\%run_name%.txt
C:\work\Tool\trec_eval\trec_eval.exe %label_path% %prediction_path%