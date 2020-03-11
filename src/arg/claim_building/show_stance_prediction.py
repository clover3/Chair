import os
import pickle

from scipy.special import softmax

from data_generator.tokenizer_wo_tf import pretty_tokens
from misc_lib import get_dir_files
from tlm.ukp.data_gen.ukp_gen_selective import load_ranked_list
from tlm.ukp.sydney_data import ukp_load_tokens_for_topic_from_shm


def job():
    topic = "abortion"
    summary_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_pred_ex_summary_w_logit"
    relevance_list_path = "/home/youngwookim/work/ukp/relevant_docs/clueweb12"
    all_tokens = ukp_load_tokens_for_topic_from_shm(topic)
    all_ranked_list = load_ranked_list(relevance_list_path)
    for file_path in get_dir_files(summary_path):
        if topic not in file_path:
            continue
        file_name = os.path.basename(file_path)
        predictions = pickle.load(open(file_path, "rb"))
        for doc_idx, preds in predictions:
            doc_id, rank, score = all_ranked_list[file_name][doc_idx]
            doc = all_tokens[doc_id]
            assert len(preds) == len(doc)
            pos_cnt = 0
            neg_cnt = 0
            for sent, pred in zip(doc, preds):
                probs = softmax(pred)
                if probs[1] > 0.5:
                    pos_cnt += 1
                elif probs[2] > 0.5:
                    neg_cnt += 1
                    print(probs, pretty_tokens(sent))
            print(pos_cnt, neg_cnt, len(doc))




if __name__ == "__main__":
    job()
