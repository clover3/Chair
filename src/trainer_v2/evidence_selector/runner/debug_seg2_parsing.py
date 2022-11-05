import os
import sys

from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.evidence_selector.evidence_candidates import PHSegmentedPairParser
from trainer_v2.train_util.arg_flags import flags_parser



def main(args):
    src_input_dir = os.path.join(job_man_dir, "evidence_candidate_gen")
    run_config = get_run_config_for_predict(args)

    input_files = os.path.join(src_input_dir, "9")
    dataset = get_classification_dataset(input_files, run_config, ModelConfig600_3(), False)
    segment_len = 300
    parser = PHSegmentedPairParser(300)
    for batch in dataset:
        x, y = batch
        input_ids, segment_ids = x
        for i in range(16):
            try:
                ret = parser.get_ph_segment_pair(input_ids[i], segment_ids[i])
            except TypeError as e:
                print(e)
                print(input_ids[i])
                print(segment_ids[i][:segment_len])
                print(segment_ids[i][segment_len:])
                exit()


if __name__ == "__main__":
    args = flags_parser.parse_args("")
    main(args)

