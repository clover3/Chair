from arg.perspectives.clueweb_helper import preload_tf, preload_docs, ClaimRankedList
from arg.perspectives.contextual_hint_analysis import get_perspective, get_relevant_unigrams, load_and_format_doc, \
    build_lm
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids
from cache import save_to_pickle
from data_generator.job_runner import WorkerInterface, JobRunner, sydney_working_dir
from list_lib import lmap


def build_single_claim_lm(all_ranked_list, claim):
    candidate_k = 50
    claim_text, perspectives = get_perspective(claim, candidate_k)
    unigrams = get_relevant_unigrams(perspectives)
    cid = claim['cId']
    ranked_list = all_ranked_list.get(str(cid))
    doc_ids = [t[0] for t in ranked_list]
    preload_docs(doc_ids)
    preload_tf(doc_ids)
    docs = lmap(load_and_format_doc, doc_ids)
    lm_classifier = build_lm(docs, unigrams)
    return lm_classifier


class ClaimLMWorker(WorkerInterface):
    def __init__(self, dummy):
        d_ids = list(load_dev_claim_ids())
        self.claims = get_claims_from_ids(d_ids)

        self.all_ranked_list = ClaimRankedList()

    def work(self, job_id):
        job_size = 10
        st = job_id * job_size
        ed = (job_id + 1) * job_size
        data = []  #
        for c in self.claims[st:ed]:
            cid = c['cId']
            lm = build_single_claim_lm(self.all_ranked_list, c)
            data.append((cid, lm))

        save_name = "dev_claim_{}".format(job_id)
        save_to_pickle(data, save_name)




if __name__ == "__main__":
    runner = JobRunner(sydney_working_dir, 20, "perspective_dev_claim_lm", ClaimLMWorker)
    runner.start()



