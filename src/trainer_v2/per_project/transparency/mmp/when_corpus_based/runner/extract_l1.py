import json
import sys

from transformers import AutoTokenizer

from job_manager.job_runner_with_server import JobRunnerS
from trainer_v2.train_util.arg_flags import flags_parser

from data_generator.job_runner import WorkerInterface
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict, RunConfig2
from trainer_v2.per_project.transparency.mmp.alignment.alignment_predictor import AlignmentPredictor, \
    compute_alignment_first_layer
from trainer_v2.per_project.transparency.mmp.alignment.grad_extractor import ModelEncoded
from trainer_v2.per_project.transparency.mmp.dev_analysis.when_term_frequency import enum_when_corpus_part
from cpath import output_path
from misc_lib import path_join
from typing import List, Iterable, Callable, Dict, Tuple, Set, Any


class Worker(WorkerInterface):
    def __init__(self,
                 run_config,
                 output_dir):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        target_q_word = "when"
        target_q_word_id = tokenizer.vocab[target_q_word]
        print("Using first layer's attention to extract alignment")

        def compute_alignment_fn(me: ModelEncoded):
            return compute_alignment_first_layer(me, target_q_word_id)

        self.align_predictor = AlignmentPredictor(run_config, compute_alignment_fn)
        self.output_dir = output_dir

        pass

    def work(self, job_id):
            def enum_qd_pairs():
                for query, doc_pos, doc_neg in enum_when_corpus_part(job_id):
                    yield query, doc_pos
                    yield query, doc_neg

            save_path = path_join(self.output_dir, str(job_id))
            num_record = 13220 / 11 * 2
            qd_itr = enum_qd_pairs()
            out_itr = self.align_predictor.predict_for_qd_iter(qd_itr, num_record)
            out_f = open(save_path, "w")
            for out_info in out_itr:
                out_f.write(json.dumps(out_info) + "\n")


def main(args):
    run_config: RunConfig2 = get_run_config_for_predict(args)
    def factory(output_dir):
         return Worker(run_config, output_dir)
    job_name = "when_tf_l1"
    root_dir = path_join(output_path, "msmarco", "passage")
    job_runner = JobRunnerS(root_dir, 11, job_name, factory)
    job_runner.auto_runner()


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

