echo off
set qrel_path=C:\work\Code\Chair\output\ca_building\qrel\0522_done.txt
set run_name=PQ_10
set prediction_path=C:\work\Code\Chair\output\ca_building\passage_ranked_list_sliced\%run_name%.txt
C:\work\Tool\trec_eval\trec_eval.exe -m map %qrel_path% %prediction_path%
set run_name=PQ_11
set prediction_path=C:\work\Code\Chair\output\ca_building\passage_ranked_list_sliced\%run_name%.txt
C:\work\Tool\trec_eval\trec_eval.exe -m map %qrel_path% %prediction_path%
set run_name=PQ_12
set prediction_path=C:\work\Code\Chair\output\ca_building\passage_ranked_list_sliced\%run_name%.txt
C:\work\Tool\trec_eval\trec_eval.exe -m map %qrel_path% %prediction_path%
set run_name=PQ_13
set prediction_path=C:\work\Code\Chair\output\ca_building\passage_ranked_list_sliced\%run_name%.txt
C:\work\Tool\trec_eval\trec_eval.exe -m map %qrel_path% %prediction_path%
