@echo off
set label_path=C:\work\Code\Chair\data\nli\mnli_ex\trec_style\all_test.txt
set run_name=tf_idf_mismatch
set prediction_path=C:\work\Code\Chair\output\mnli_ex\ranked_list\%run_name%.txt
C:\work\Tool\trec_eval\trec_eval.exe -m map %label_path% %prediction_path%
C:\work\Tool\trec_eval\trec_eval.exe -m num_q %label_path% %prediction_path%