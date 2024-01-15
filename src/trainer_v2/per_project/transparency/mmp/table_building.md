

# Table Eval

{common} = src/trainer_v2/per_project/transparency/mmp


## Table Inference

* ${common}/pep_to_tt/runner/score_pairs.py {conf} {slurm_job_idx}

## Combine Table

* ${common}/pep/term_pair_scoring_runner/combination_filtering_constant.py