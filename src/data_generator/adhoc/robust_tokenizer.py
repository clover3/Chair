import os
import pickle
import re

from data_generator.common import get_tokenizer
from data_generator.data_parser import trec
from data_generator.data_parser.robust2 import load_qrel, load_bm25_best
from data_generator.job_runner import sydney_working_dir
from misc_lib import TimeEstimator


class RobustPreprocess:
    def __init__(self):
        robust_path = "/mnt/nfs/work3/youngwookim/data/robust04"
        self.data = trec.load_robust(robust_path)

    def tokenize_docs(self, doc_id_list):
        tokenizer = get_tokenizer()
        token_d = {}
        ticker = TimeEstimator(len(doc_id_list))
        for doc_id in doc_id_list:
            text = self.data[doc_id]
            text = re.sub(r"<\s*[^>]*>", " ", text)
            # tokenize text
            tokens = tokenizer.tokenize(text)
            token_d[doc_id] = tokens
            ticker.tick()

        return token_d


class RobustPreprocessTrain(RobustPreprocess):
    def __init__(self):
        super(RobustPreprocessTrain, self).__init__()
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrel(qrel_path)

    def tokenize(self, job_id):
        all_doc_id_list = []
        for query in self.judgement.keys():
            judgement = self.judgement[query]
            for doc_id, score in judgement.items():
                all_doc_id_list.append(doc_id)

        n_total_docs = 174787
        interval = 10000

        all_doc_id_list = list(set(all_doc_id_list))
        all_doc_id_list.sort()
        st = interval * job_id
        ed = interval * (job_id + 1)
        target_doc_list = all_doc_id_list[st:ed]
        if not target_doc_list:
            return
        return self.tokenize_docs(target_doc_list)


class RobustPreprocessPredict(RobustPreprocess):
    def __init__(self, top_k=150):
        super(RobustPreprocessPredict, self).__init__()
        self.galago_rank = load_bm25_best()
        self.top_k = top_k

    def get_required_doc_ids(self):
        doc_id_set = set()
        for query_id, ranked_list in self.galago_rank.items():
            ranked_list.sort(key=lambda x:x[1])
            doc_id_set.update([x[0] for x in ranked_list[:self.top_k]])
        return list(doc_id_set)

    def tokenize(self):
        all_doc_id_list = self.get_required_doc_ids()
        all_doc_id_list.sort()
        return self.tokenize_docs(all_doc_id_list)



if __name__ == "__main__":
    gen = RobustPreprocessPredict()
    d = gen.tokenize()

    save_path = os.path.join(sydney_working_dir, "RobustPredictTokens3", "1")
    pickle.dump(d, open(save_path, "wb"))
