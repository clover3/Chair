from collections import Counter

###
import datastore.tool
from cache import save_to_pickle
from data_generator.job_runner import JobRunner, sydney_working_dir
from datastore.interface import get_sliced_rows
from datastore.table_names import TokenizedCluewebDoc, CluewebDocTF
from misc_lib import TimeEstimator


class Worker:
    def __init__(self, out_path_not_used):
        pass

    def work(self, job_id):
        buffered_saver = datastore.tool.PayloadSaver()
        job_size = 10000
        st = job_id * job_size
        ed = (job_id+1) * job_size
        rows = get_sliced_rows(TokenizedCluewebDoc, st , ed)
        acc_count = Counter()
        ticker = TimeEstimator(job_size)
        for key, value in rows:
            tokens = value
            count = Counter(tokens)
            acc_count.update(count)
            buffered_saver.save(CluewebDocTF, key, count)
            ticker.tick()

        save_name = "tf_payload_{}".format(job_id)
        save_to_pickle(buffered_saver, save_name)
        save_name = "acc_count_{}".format(job_id)
        save_to_pickle(acc_count, save_name)


if __name__ == "__main__":
    runner = JobRunner(sydney_working_dir, 121, "pc_tf_saver", Worker)
    runner.start()


