import os
import pickle
from collections import Counter

import data_generator.argmining.ukp_header
from data_generator.job_runner import JobRunner, sydney_working_dir
from galagos.parse import load_galago_ranked_list
from misc_lib import get_dir_files
from tlm.data_gen.base import UnmaskedPairedDataGen
from tlm.ukp.data_gen.run_ukp_gen1 import UkpWorker
from tlm.ukp.sydney_data import sydney_get_ukp_ranked_list


class UkpSelectiveWorker(UkpWorker):
    def __init__(self, out_path, top_k, generator, selector):
        super(UkpSelectiveWorker, self).__init__(out_path, top_k, generator, 0)
        self.selector = selector

    def work(self, job_id):
        topic = data_generator.argmining.ukp_header.all_topics[job_id]
        ranked_list = sydney_get_ukp_ranked_list()[topic]
        ranked_list = self.selector(ranked_list)
        doc_ids = [doc_id for doc_id, _, _ in ranked_list[self.drop_head:self.top_k]]
        all_tokens = self.load_tokens_for_topic(topic)

        docs = [all_tokens[doc_id] for doc_id in doc_ids if doc_id in all_tokens]
        insts = self.generator.create_instances_from_documents(docs)
        output_file = os.path.join(self.out_dir, topic.replace(" ", "_"))
        self.generator.write_instances(insts, output_file)


def load_ranked_list(relevance_list_path):
    all_ranked_list = {}
    for file_path in get_dir_files(relevance_list_path):
        file_name = os.path.basename(file_path)
        ranked_list_d = load_galago_ranked_list(file_path)

        queries = ranked_list_d.keys()
        any_query = list(queries)[0]
        ranked_list = ranked_list_d[any_query]
        all_ranked_list[file_name] = ranked_list
    return all_ranked_list


class SelectByLabelScore:
    def __init__(self, select_by_preds):
        summary_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_pred_summary"
        relevance_list_path = "/home/youngwookim/work/ukp/relevant_docs/clueweb12"
        all_ranked_list = load_ranked_list(relevance_list_path)

        self.selected = set()
        for file_path in get_dir_files(summary_path):
            file_name = os.path.basename(file_path)
            predictions = pickle.load(open(file_path, "rb"))
            n_reject = 0
            for doc_idx, preds in predictions:
                doc_id, rank, score = all_ranked_list[file_name][doc_idx]
                assert rank == doc_idx + 1
                if select_by_preds(preds):
                    self.selected.add(doc_id)
                else:
                    n_reject += 1
            print("{} Reject {}".format(file_name, n_reject / len(predictions)))

    def select(self, ranked_list):
        r = []
        for e in ranked_list:
            doc_id, _, _ = e
            if doc_id in self.selected:
                r.append(e)

        return r


def preds_to_select(preds):
    c = Counter(preds)
    non_argument =  c[0] / sum(c.values())
    return non_argument < 0.9


if __name__ == "__main__":
    top_k = 150000
    num_jobs = len(data_generator.argmining.ukp_header.all_topics) - 1
    generator = UnmaskedPairedDataGen()
    selector = SelectByLabelScore(preds_to_select)
    JobRunner(sydney_working_dir, num_jobs, "ukp_drop_0",
              lambda x: UkpSelectiveWorker(x, top_k, generator, selector.select)).start()


