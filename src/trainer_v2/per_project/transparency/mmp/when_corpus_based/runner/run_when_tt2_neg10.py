import os
import random
from typing import Iterable, Tuple

from cpath import output_path
from data_generator.job_runner import WorkerInterface
from dataset_specific.msmarco.passage.passage_resource_loader import MMPPosNegSampler, tsv_iter, enum_grouped
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.mmp.data_gen.tt2_train_gen import get_encode_fn_when_bow
from trainer_v2.per_project.transparency.mmp.when_corpus_based.feature_encoder import BM25TFeatureEncoder
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import get_mmp_bm25, get_candidate_voca


class Worker(WorkerInterface):
    def __init__(self, out_dir):
        bm25 = get_mmp_bm25()
        candidate_voca = get_candidate_voca()
        feature_encoder = BM25TFeatureEncoder(bm25, candidate_voca)
        fn = feature_encoder.get_term_translation_weight_feature
        self.out_dir = out_dir
        self.encode_fn = get_encode_fn_when_bow(fn)
        self.pos_neg_sampler = MMPPosNegSampler()
        self.n_neg = 10

    def enum_pos_neg(self, job_id):
        file_path = path_join(output_path, "msmarco", "passage", "when_full_re", str(job_id))
        itr = tsv_iter(file_path)
        itr2 = enum_grouped(itr)

        for group in itr2:
            try:
                pos_docs, neg_docs = self.pos_neg_sampler.split_pos_neg_entries(group)
                for pos_doc in pos_docs:
                    qid, pid, query, pos_text = pos_doc
                    random.shuffle(neg_docs)
                    for neg_doc in neg_docs[:self.n_neg]:
                        qid, pid, query, neg_text = neg_doc
                        yield query, pos_text, neg_text
            except IndexError:
                pass

    def work(self, job_no):
        save_path = os.path.join(self.out_dir, str(job_no))
        itr: Iterable[Tuple[str, str, str]] = self.enum_pos_neg(job_no)
        write_records_w_encode_fn(save_path, self.encode_fn, itr)


def main():
    working_dir = path_join("output", "msmarco", "passage")
    runner = JobRunnerS(working_dir, 17, "train_when_tt2", Worker)
    runner.start()



if __name__ == "__main__":
    main()