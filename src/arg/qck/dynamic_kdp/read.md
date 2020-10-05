

1. requestor.py : Sydney
2. start_kdp_eval_server.py : GCloud
    output/request_dir
    output/request_dir/req_job_info.json
    

3. tfrecord_maker.py
 - This process write to task/task_sh/1.sh

 
4. task_manager/task_executor.py
  + cppnc_auto.sh
  - task/task_log/info.json
  - output_cppnc/1.score
5. score_summarizer.py
  - output_cppnc/1.summary
  - info state : output/request_dir/score_summarizer_job_info.json
  
6. dvp_visualize.py

