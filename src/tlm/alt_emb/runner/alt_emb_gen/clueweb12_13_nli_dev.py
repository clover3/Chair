import os

from cache import load_from_pickle
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.alt_emb.add_alt_emb import MatchTree, convert_alt_emb2


class Worker:
    def __init__(self, match_tree: MatchTree, out_path):
        self.out_dir = out_path
        self.match_tree = match_tree

    def work(self, job_id):
        input_path = os.path.join(sydney_working_dir, "clueweb12_13B_no_short", str(job_id))
        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        convert_alt_emb2(input_path, output_file, self.match_tree)


if __name__ == "__main__":
    def worker_gen(out_dir):
        match_tree = load_from_pickle("match_tree_nli_dev")
        return Worker(match_tree, out_dir)

    runner = JobRunner(sydney_working_dir, 1000, "alt_emb_13", worker_gen)
    runner.start()



