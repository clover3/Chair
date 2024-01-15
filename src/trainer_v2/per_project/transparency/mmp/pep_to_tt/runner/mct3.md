



* Purpose: Simplified term-pair evaluation, which only consider promising candidates. 
* Candidates are from pep_tt3's prediction.  

* ${common}: src/trainer_v2/per_project/transparency/mmp
* ${common}/pep_to_tt/runner/score_mct3.py ${model} ${step} ${job_idx}

* ${common}/pep_to_tt/runner/combine_mct3.py ${model} ${step}

* Script that build a table
* unity: bash mmp/submit_mct_table_build.sh ${step}
* sydney: 
  * bash sync_table.sh 
  * python src/trainer_v2/per_project/transparency/mmp/retrieval_run/table_benchmark.py output/mmp/tables/mtc3_pep_tt7_20000.tsv mtc3_pep_tt7_20K