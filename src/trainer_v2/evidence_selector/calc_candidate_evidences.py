import os
import pickle
from typing import List, Iterator

from data_generator.job_runner import WorkerInterface
from misc_lib import TimeEstimator
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.inference import InferenceHelperSimple
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model_n_label_3
from trainer_v2.evidence_selector.evidence_candidates import PHSegmentedPairParser, ScoredEvidencePair


class EvidenceCompare:
    def __init__(self, parser: PHSegmentedPairParser):
        self.parser = parser

    def enum_grouped(self, batch_prediction_enum) -> Iterator[List[ScoredEvidencePair]]:
        group: List[ScoredEvidencePair] = []
        for x, output in batch_prediction_enum:
            input_ids, segment_ids = x
            batch_size, _ = input_ids.shape
            l_y, g_y_ = output
            assert len(g_y_) == 1
            g_y = g_y_[0]

            for i in range(batch_size):
                pair = self.parser.get_ph_segment_pair(input_ids[i], segment_ids[i])
                e = ScoredEvidencePair(pair, g_y[i], l_y[i])

                if pair.is_base_inst() and group:
                    yield group
                    group = []
                group.append(e)

        if group:
            yield group


class Worker(WorkerInterface):
    def __init__(self, run_config, src_input_dir, output_dir):
        model = load_local_decision_model_n_label_3(run_config.predict_config.model_save_path)
        self.inference = InferenceHelperSimple(model)
        self.output_dir = output_dir
        self.src_input_dir = src_input_dir
        self.run_config = run_config
        parser = PHSegmentedPairParser(300)
        self.evidence_compare = EvidenceCompare(parser)

    def work(self, job_id):
        save_path = os.path.join(self.output_dir, str(job_id))

        def build_dataset(input_files, is_for_training):
            return get_classification_dataset(input_files, self.run_config, ModelConfig600_3(), is_for_training)

        eval_file_path = os.path.join(self.src_input_dir, str(job_id))
        predict_dataset = build_dataset(eval_file_path, False)
        n_group_maybe = 10000

        batch_prediction_enum = self.inference.enum_batch_prediction(predict_dataset)
        ticker = TimeEstimator(n_group_maybe)
        output = []
        for group in self.evidence_compare.enum_grouped(batch_prediction_enum):
            output.append(group)
            ticker.tick()

        pickle.dump(output, open(save_path, "wb"))

