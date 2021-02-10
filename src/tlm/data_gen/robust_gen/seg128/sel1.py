from functools import partial

from arg.robust.qc_gen import RobustWorker
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import AllSegmentAsDoc, RobustTrainGenSelected2
from tlm.data_gen.robust_gen.select_supervision.selection_fns import get_selection_fn1


def main():
    max_passage_length = 128
    encoder = AllSegmentAsDoc(max_passage_length)
    target_selection_fn = get_selection_fn1(0.5)
    max_seq_length = max_passage_length
    worker_factory = partial(RobustWorker, RobustTrainGenSelected2(encoder, max_seq_length, "desc", target_selection_fn))
    runner = JobRunner(job_man_dir, 4, "sel128_1", worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
