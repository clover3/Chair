import os
import pickle
from typing import List

from arg.claim_building.clueweb12_B13_termstat import load_clueweb12_B13_termstat
from arg.perspectives.basic_analysis import load_train_data_point
from arg.perspectives.ranked_list_interface import Q_CONFIG_ID_BM25_10000, StaticRankedListInterface
from arg.perspectives.select_paragraph import select_paragraph_dp_list, ParagraphClaimPersFeature
from data_generator.common import get_tokenizer
from data_generator.job_runner import JobRunner, sydney_working_dir


class Worker:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.ci = StaticRankedListInterface(Q_CONFIG_ID_BM25_10000)
        self.all_data_points = load_train_data_point()
        _, clue12_13_df = load_clueweb12_B13_termstat()
        self.clue12_13_df = clue12_13_df
        self.tokenizer = get_tokenizer()

    def work(self, job_id):
        step = 10
        st = job_id * step
        ed = (job_id + 1) * step

        print("load_train_data_point")

        print("select paragraph")
        todo = self.all_data_points[st:ed]
        features: List[ParagraphClaimPersFeature] = select_paragraph_dp_list(self.ci, self.clue12_13_df, self.tokenizer, todo)
        pickle.dump(features, open(os.path.join(self.out_dir, str(job_id)), "wb"))


if __name__ == "__main__":
    runner = JobRunner(sydney_working_dir, 1000, "perspective_paragraph_feature", Worker)
    runner.start()
