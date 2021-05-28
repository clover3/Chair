from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import train_query_group_len
from epath import job_man_dir
from tlm.data_gen.msmarco_doc_gen.fast_gen.best_seg_prediction_gen import BestSegmentPredictionGen


def main():
    split = "train"

    def factory(out_dir):
        return BestSegmentPredictionGen(
            512,
            split,
            skip_single_seg=False,
            pick_for_pairwise=True,
            out_dir=out_dir)

    runner = JobRunner(job_man_dir, train_query_group_len - 1,
                       "MMD_best_seg_prediction_{}_A".format(split), factory)
    runner.start()


if __name__ == "__main__":
    main()
