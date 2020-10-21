working_path ="/mnt/nfs/work3/youngwookim/data/tlm_simple"
from cache import *
from cpath import data_path
from data_generator import tokenizer_wo_tf as tokenization
from galagos.parse import load_galago_judgement2
from misc_lib import TimeEstimator
from sydney_manager import ReadyMarkTaskManager
from tlm.two_seg_pretraining import write_predict_instance
from tlm.wiki.tf_instance_maker import TFInstanceMaker
from tlm.wiki.token_db import load_seg_token_readers


class Worker:
    def __init__(self):
        self.token_reader = load_seg_token_readers()
        vocab_file = os.path.join(data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        vocab = vocab_words = list(self.tokenizer.vocab.keys())

        self.tf_inst_maker = TFInstanceMaker(vocab)

    def work(self, job_id):
        max_pred = 20
        max_seq_length = 512
        problem_per_job = 100 * 1000
        problem_path = os.path.join(working_path, "problems", "{}".format(job_id))
        query_res_path = os.path.join(working_path, "q_res", "{}.txt".format(job_id))
        output_path = os.path.join(working_path, "tf", "tf_{}".format(job_id))

        problems = pickle.load(open(problem_path, "rb"))
        query_res = load_galago_judgement2(query_res_path)

        named_instanace_list = []
        missing = 0
        ticker = TimeEstimator(len(problems))
        for idx, problem in enumerate(problems):
            qid = str(job_id * problem_per_job + idx)
            if qid not in query_res:
                missing += 1
                continue

            target_tokens, prev_seg, cur_seg, next_seg, prev_tokens, next_tokens, mask_indice, doc_id = problem

            # get top rank doc
            rank_list = query_res[qid]
            for i in range(len(rank_list)):
                e_doc_id, e_rank, _ = rank_list[i]
                if e_doc_id == doc_id:
                    rank_list = rank_list[:i] + rank_list[i + 1:]
                    break

            if not rank_list:
                continue

            e_doc_id , e_rank, _ = rank_list[0]

            # get segment content
            hint_tokens = self.token_reader.load(e_doc_id)
            tokenization._truncate_seq_pair(target_tokens, hint_tokens, max_seq_length - 3)
            # combine segments
            prob = target_tokens, hint_tokens, mask_indice
            inst = self.tf_inst_maker.make_instance(prob)
            unique_id = int(qid)
            named_instanace_list.append((inst, unique_id))
            ticker.tick()
        if missing > 0.1 * len(problems):
            print("Too many missing : {} of {}".format(missing, len(problems)))
        write_predict_instance(named_instanace_list, self.tokenizer, max_seq_length, max_pred, [output_path])



def main():
    mark_path = os.path.join(working_path, "p2r_3_mark")
    ready_path = os.path.join(working_path, "q_res", "{}.txt")

    mtm = ReadyMarkTaskManager(1000, ready_path, mark_path)

    worker = Worker()
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)

if __name__ == "__main__":
    main()
