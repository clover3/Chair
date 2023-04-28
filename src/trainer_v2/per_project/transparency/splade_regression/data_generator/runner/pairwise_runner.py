import os
from cpath import output_path
from misc_lib import path_join, ceil_divide

from transformers import AutoTokenizer

from data_generator.job_runner import WorkerInterface
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.splade_regression.data_loaders.pairwise_eval import load_pairwise_mmp_data
from trainer_v2.per_project.transparency.transformers_utils import get_multi_text_encode_fn


class Worker(WorkerInterface):
    def __init__(self, output_dir):
        checkpoint_model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_model_name)
        self.output_dir = output_dir

    def work(self, job_id):
        job_size = 10
        st = job_id * job_size
        ed = st + job_size
        target_partition = list(range(st, ed))
        triplet_list = load_pairwise_mmp_data(target_partition)
        tokenizer = self.tokenizer

        def tokenize_triplet(triplet):
            t1, t2, t3 = triplet
            return tokenizer(t1), tokenizer(t2), tokenizer(t3)

        itr = map(tokenize_triplet, triplet_list)
        max_seq_length = 256
        save_path = os.path.join(self.output_dir, f"{job_id}")
        encode_fn = get_multi_text_encode_fn(max_seq_length, n_text=3)
        n_item = 10000 * job_size
        write_records_w_encode_fn(save_path, encode_fn, itr, n_item)


def main():
    max_job = ceil_divide(5279, 10)
    root_dir = path_join(output_path, "msmarco", "passage")
    job_name = "pairwise_train"
    runner = JobRunnerS(root_dir, max_job, job_name, Worker)
    runner.auto_runner()


if __name__ == "__main__":
    main()
