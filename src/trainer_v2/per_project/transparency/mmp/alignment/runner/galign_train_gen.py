import os
import random
import sys
from collections import OrderedDict, Counter
from cpath import output_path
from misc_lib import path_join, get_second, pick1

from transformers import AutoTokenizer
from typing import List, Iterable, Callable, Dict, Tuple, Set

from data_generator.create_feature import create_int_feature
from data_generator.job_runner import WorkerInterface
from dataset_specific.msmarco.passage.passage_resource_loader import MMPPosNegSampler, enum_grouped, tsv_iter
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.alignment.galign_label import compute_gain_10K_when

NI = NotImplemented


def get_encode_fn_for_galign_paired(q_term, d_terms_pos, d_terms_neg, counter=None):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    q_term_id = tokenizer.vocab[q_term]

    def enc_d_term(d_terms):
        skipped_terms = []
        term_id_list = []
        for t in d_terms:
            try:
                token_id = tokenizer.vocab[t]
                term_id_list.append(token_id)
            except KeyError as e:
                skipped_terms.append(t)
        # d_term_id_list = [tokenizer.vocab[t] for t in d_terms]
        d_term_id_set = set(term_id_list)
        print("skipped : ", skipped_terms)
        return d_term_id_set

    d_terms_pos_ids = enc_d_term(d_terms_pos)
    d_terms_neg_ids = enc_d_term(d_terms_neg)
    print("Pos/neg terms: {}/{} ".format(len(d_terms_pos_ids), len(d_terms_neg_ids)))

    def add_count(event):
        counter[event] += 1

    def encode_text_pair(query, document):
        encoded_input = tokenizer.encode_plus(
            query,
            document,
            padding="max_length",
            max_length=256,
            truncation=True,
        )

        input_ids = encoded_input["input_ids"]
        token_type_ids = encoded_input["token_type_ids"]

        q_term_mask = [0] * len(input_ids)
        d_term_mask = [0] * len(input_ids)

        pos_match_indices = []
        neg_match_indices = []
        for i in range(len(input_ids)):
            is_query = token_type_ids[i] == 0
            if is_query and input_ids[i] == q_term_id:
                q_term_mask[i] = 1
                add_count("q_term")
            else:
                if input_ids[i] in d_terms_pos_ids:
                    pos_match_indices.append(i)
                elif input_ids[i] in d_terms_neg_ids:
                    neg_match_indices.append(i)

        if pos_match_indices and neg_match_indices:
            do_pos = random.randint(0, 1)
            do_neg = not do_pos
        elif pos_match_indices:
            do_pos = True
            do_neg = False
        elif neg_match_indices:
            do_pos = False
            do_neg = True
        else:
            do_pos = False
            do_neg = False

        if do_neg:
            i = pick1(neg_match_indices)
            d_term_mask[i] = 1
            add_count("d_term_neg")
            label = 0
            is_valid = 1
        elif do_pos:
            i = pick1(pos_match_indices)
            d_term_mask[i] = 1
            add_count("d_term_pos")
            label = 1
            is_valid = 1
        else:
            label = 0
            is_valid = 0

        attention_mask = encoded_input["attention_mask"]
        return input_ids, token_type_ids, q_term_mask, d_term_mask, label, is_valid

    def encode_fn(q_pos_neg: Tuple[str, str, str]):
        q, d1, d2 = q_pos_neg
        feature: OrderedDict = OrderedDict()
        for i, d in [(1, d1), (2, d2)]:
            input_ids, token_type_ids, q_term_mask, d_term_mask, label, is_valid = encode_text_pair(q, d)
            feature[f"input_ids{i}"] = create_int_feature(input_ids)
            feature[f"token_type_ids{i}"] = create_int_feature(token_type_ids)
            feature[f"q_term_mask{i}"] = create_int_feature(q_term_mask)
            feature[f"d_term_mask{i}"] = create_int_feature(d_term_mask)
            feature[f"label{i}"] = create_int_feature([label])
            feature[f"is_valid{i}"] = create_int_feature([is_valid])
        return feature
    return encode_fn


class Worker(WorkerInterface):
    def __init__(self, encode_fn, out_dir):
        self.out_dir = out_dir
        self.encode_fn = encode_fn
        self.pos_neg_sampler = MMPPosNegSampler()

    def enum_pos_neg(self, job_id):
        file_path = path_join(output_path, "msmarco", "passage", "when_full_re", str(job_id))
        itr = tsv_iter(file_path)
        itr2 = enum_grouped(itr)

        for group in itr2:
            try:
                pos_doc, neg_doc = self.pos_neg_sampler.sample_pos_neg(group)
                qid, pid, query, pos_text = pos_doc
                qid, pid, query, neg_text = neg_doc
                yield query, pos_text, neg_text
            except IndexError:
                pass

    def work(self, job_no):
        save_path = os.path.join(self.out_dir, str(job_no))
        itr: Iterable[Tuple[str, str, str]] = self.enum_pos_neg(job_no)
        write_records_w_encode_fn(save_path, self.encode_fn, itr, 1000)


def main():
    term_gain = compute_gain_10K_when()
    term_gain.sort(key=get_second)
    # This term gain is rank change. Lower (negative) the better
    pos_terms = []
    for term, score in term_gain[:1000]:
        if score < 0:
            pos_terms.append(term)

    neg_terms = []
    for term, score in term_gain[-1000:]:
        if score > 0:
            neg_terms.append(term)
    target_q_term = "when"

    job_no = int(sys.argv[1])
    counter = Counter()
    encode_fn = get_encode_fn_for_galign_paired(target_q_term, pos_terms, neg_terms, counter)
    output_dir = path_join(output_path, "msmarco", "passage", "when_full_re_galign")
    worker = Worker(encode_fn, output_dir)
    worker.work(job_no)
    print(counter)


if __name__ == "__main__":
    main()