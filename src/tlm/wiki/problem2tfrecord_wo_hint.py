working_path ="/mnt/nfs/work3/youngwookim/data/tlm_simple"
import random

from cache import *
from data_generator import tokenizer_wo_tf as tokenization
from misc_lib import TimeEstimator
from path import data_path
from sydney_manager import MarkedTaskManager
from tlm.two_seg_pretraining import write_predict_instance
from tlm.wiki.tf_instance_maker import TFInstanceMakerPair
from tlm.wiki.token_db import load_seg_token_readers


class Worker:
    def __init__(self):
        self.token_reader = load_seg_token_readers()
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        vocab = vocab_words = list(self.tokenizer.vocab.keys())

        self.tf_inst_maker = TFInstanceMakerPair(vocab)

    def work(self, job_id):
        max_pred = 40
        max_seq_length = 512
        problem_per_job = 100 * 1000
        problem_path = os.path.join(working_path, "problems", "{}".format(job_id))
        query_res_path = os.path.join(working_path, "q_res", "{}.txt".format(job_id))
        output_path = os.path.join(working_path, "tf_wo_hint", "tf_{}".format(job_id))

        problems = pickle.load(open(problem_path, "rb"))
        random.shuffle(problems)

        named_instanace_list = []
        missing = 0
        ticker = TimeEstimator(len(problems))
        idx =0
        while idx + 1 < len(problems):
            qid = str(job_id * problem_per_job + idx)

            problem1 = problems[idx]
            problem2 = problems[idx+1]
            target_tokens_1, prev_seg, cur_seg, next_seg, prev_tokens, next_tokens, mask_indice_1, doc_id = problem1
            target_tokens_2, prev_seg, cur_seg, next_seg, prev_tokens, next_tokens, mask_indice_2, doc_id = problem2

            # get top rank doc

            # get segment content
            tokenization._truncate_seq_pair(target_tokens_1, target_tokens_2, max_seq_length - 3)
            # combine segments
            inst = self.tf_inst_maker.make_instance(target_tokens_1, mask_indice_1, target_tokens_2, mask_indice_2)
            unique_id = int(qid)
            named_instanace_list.append((inst, unique_id))
            ticker.tick()
            idx += 1
        if missing > 0.1 * len(problems):
            print("Too many missing : {} of {}".format(missing, len(problems)))
        write_predict_instance(named_instanace_list, self.tokenizer, max_seq_length, max_pred, [output_path])



def main():
    mark_path = os.path.join(working_path, "p2r_wo_hint_mark")

    mtm = MarkedTaskManager(1000, mark_path, 1)
    worker = Worker()
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)

if __name__ == "__main__":
    main()
