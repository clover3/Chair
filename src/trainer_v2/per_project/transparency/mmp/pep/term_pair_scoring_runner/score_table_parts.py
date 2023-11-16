import sys

from omegaconf import OmegaConf

from table_lib import tsv_iter
from taskman_client.job_group_proxy import SubJobContext
from trainer_v2.per_project.transparency.misc_common import read_term_pair_table
from trainer_v2.per_project.transparency.mmp.pep.inf_helper import predict_with_fixed_context_model_and_save
from cpath import output_path
from misc_lib import path_join


def main():
    conf_path = sys.argv[1]
    job_no = int(sys.argv[2])
    conf = OmegaConf.load(conf_path)
    candidate_pairs = read_term_pair_table(conf.table_path)
    num_items = len(candidate_pairs)
    job_size = int(conf.job_size)
    job_name = conf.run_name
    max_job = (num_items + job_size - 1) // job_size
    st = job_no * job_size
    ed = st + job_size
    todo = candidate_pairs[st: ed]
    if not todo:
        print("No job")
        return
    model_path = conf.model_path
    log_path = path_join(conf.save_dir, f"{job_no}.txt")
    with SubJobContext(job_name, job_no, max_job):
        predict_with_fixed_context_model_and_save(model_path, log_path, todo, 100, len(todo))


if __name__ == "__main__":
    main()


# 1K/per min