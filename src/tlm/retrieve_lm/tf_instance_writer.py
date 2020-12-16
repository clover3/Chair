import collections
import os
import pickle
import random

import cpath
from cache import load_pickle_from
from data_generator import tokenizer_wo_tf as tokenization
from list_lib import flatten
from job_manager.marked_task_manager import MarkedTaskManager
from tlm.retrieve_lm.robust_tokens import load_robust_token


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                             is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
                [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
                [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
                [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def get_candiset(i):
    p = os.path.join(cpath.data_path, "stream_pickled", "CandiSet_{}_0".format(i))

    if not os.path.exists(p):
        return None

    return load_pickle_from(p)

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",["index", "label"])

class TFRecordMaker:
    def __init__(self, max_seq):
        print("TFRecordMaker Init")
        self.max_seq = max_seq
        self.robust_tokens = load_robust_token()
        vocab_file = os.path.join(cpath.data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)
        self.vocab_words = list(self.tokenizer.vocab.keys())
        self.rng = random.Random(0)
        def load_pickle(name):
            p = os.path.join(cpath.data_path, "adhoc", name + ".pickle")
            return pickle.load(open(p, "rb"))

        self.seg_info = load_pickle("robust_seg_info")
        print("TFRecordMaker Init Done")

    def get_prev_candi_tf(self, e):
        target = e["target_tokens"]
        hint = e["prev_tokens"]
        return target, hint

    def get_segment(self, doc_id, target_loc):
        interval_list = self.seg_info[doc_id]
        target_loc_ed = None
        for e in interval_list:
            (loc, loc_ed), (loc_sub, loc_sub_ed) = e
            if loc == target_loc:
                target_loc_ed = loc_ed
                break

        if target_loc_ed is None:
            print("Invalid LOC")
            print("doc_id : ", doc_id)
            print("target_loc : ", target_loc)
            print("interval_list : ", interval_list)
            raise Exception()

        tokens = self.robust_tokens[doc_id]
        seg_tokens = tokens[target_loc:target_loc_ed]

        return flatten([self.tokenizer.wordpiece_tokenizer.tokenize(t) for t in seg_tokens])

    def get_subtoken_problem_local(self, e):
        cut = 256 - 3
        p1 = e["target_tokens"], e["prev_tokens"][:cut], e["mask_indice"]
        p2 = e["target_tokens"], e["next_tokens"][:cut], e["mask_indice"]
        return p1, p2

    def random_voca(self):
        return self.vocab_words[self.rng.randint(0, len(self.vocab_words) - 1)]

    def make_instance(self, problem):
        target, hint, mask = problem
        rng = self.rng
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in target:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in hint:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        output_tokens = list(tokens)
        masked_lms = []

        for index in mask:
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = self.random_voca()
            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        instance = TrainingInstance(
            tokens=output_tokens,
            segment_ids=segment_ids,
            is_random_next=False,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        return instance

    def get_problem_remote(self, e, p_id):
        target = e["target_tokens"]
        passages = e["passages"]
        doc_id, loc = passages[p_id]
        hint = self.get_segment(doc_id, loc)
        return target, hint, e["mask_indice"]

    def generate_ir_tfrecord(self, e):
        p1,p2 = self.get_subtoken_problem_local(e)
        l = [p1,p2]

        n_passage = len(e["passages"])

        for i in range(n_passage):
            p = self.get_problem_remote(e, i)
            l.append(p)

        return l

def worker(tf_maker, job_id):
    cs = get_candiset(job_id)
    if cs is None:
        return

    info_list = []
    inst_list = []


    maker = tf_maker.generate_ir_tfrecord

    for j, e in enumerate(cs):
        candi_id = "{}_{}".format(job_id, j)
        for idx, problem in enumerate(maker(e)):
            tf_id = candi_id + "_{}".format(idx)
            inst = tf_maker.make_instance(problem)

            info_list.append(tf_id)
            inst_list.append(inst)

    result = inst_list, info_list

    p = os.path.join(cpath.data_path , "tlm", "instances_local", "inst_{}.pickle".format(job_id))
    pickle.dump(result, open(p, "wb"))



def main():
    max_seq = 512
    print("TF_Instance_writer")
    tf_maker = TFRecordMaker(max_seq)

    mark_path = os.path.join(cpath.data_path, "tlm", "tf_inst_local_mark")
    mtm = MarkedTaskManager(1000*1000, mark_path, 1000)
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker(tf_maker, job_id)
        job_id = mtm.pool_job()

if __name__ == "__main__":
    main()
