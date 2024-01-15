from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.datagen2.table_lookup_common import load_table_per_worker, \
    table_lookup
import sys
from omegaconf import OmegaConf


def table_lookup_val(conf, worker_no):
    table_d = load_table_per_worker(conf, worker_no)
    c_log.info("Iterating Todos")
    job_no = 0
    file_name = f"{job_no}_{worker_no}"
    todo_path = path_join(conf.val_lookup_todo_dir, file_name)
    save_path = path_join(conf.val_lookup_output_dir, file_name)
    table_lookup(todo_path, save_path, table_d)


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    worker_no = int(sys.argv[2])
    with JobContext(f"TableLookup_{worker_no}"):
        table_lookup_val(conf, worker_no)


if __name__ == "__main__":
    main()