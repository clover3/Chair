import sys

from omegaconf import OmegaConf

from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.datagen2.table_lookup_common import load_table_per_worker, \
    table_lookup


def table_lookup_train_split(conf, worker_no):
    lookup_todo_dir: str = conf.lookup_todo_dir
    lookup_output_dir = conf.lookup_output_dir
    table_d = load_table_per_worker(conf, worker_no)
    c_log.info("Iterating Todos")
    for job_no in range(20):
        file_name = f"{job_no}_{worker_no}"
        todo_path = path_join(lookup_todo_dir, file_name)
        save_path = path_join(lookup_output_dir, file_name)
        table_lookup(todo_path, save_path, table_d)


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    worker_no = int(sys.argv[2])
    with JobContext(f"TableLookup_{worker_no}"):
        table_lookup_train_split(conf, worker_no)


if __name__ == "__main__":
    main()