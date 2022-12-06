import os
import random
from typing import List, Tuple, Callable

import tensorflow as tf

import cpath
from cache import load_pickle_from, save_list_to_jsonl_w_fn
from data_generator.NLI.nli_info import nli_tokenized_path
from data_generator.job_runner import WorkerInterface
from dataset_specific.mnli.mnli_reader import MNLIReader
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import ceil_divide, TimeEstimator
from port_info import LOCAL_DECISION_PORT
from trainer.promise import PromiseKeeper, MyPromise
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.inference import InferenceHelper
from trainer_v2.custom_loop.per_task.nli_ts_util import load_local_decision_model, dataset_factory_600_3
from trainer_v2.keras_server.nlits_client import NLITSClient
from trainer_v2.per_project.cip.cip_common import get_random_split_location, split_into_two, \
    SegmentationTrialInputs, SegmentationTrials
from trainer_v2.per_project.cip.nlits_direct import TS600_3_Encoder, reslice_local_global_decisions
from trainer_v2.per_project.cip.path_helper import get_nlits_segmentation_trial_save_path, \
    get_nlits_segmentation_trial_subjob_save_dir
from trainer_v2.train_util.get_tpu_strategy import get_strategy2


def try_segmentations_and_save(
        nltis_server_addr,
        base_seq_length,
):
    split = "train"
    reader = MNLIReader()
    query_batch_size = 64
    num_step = ceil_divide(reader.get_data_size(split), query_batch_size)
    ticker = TimeEstimator(num_step)
    data: List[Tuple[List[int], List[int], int]] = load_pickle_from(nli_tokenized_path(split))
    nlits_client: NLITSClient = NLITSClient(nltis_server_addr, LOCAL_DECISION_PORT, base_seq_length)
    predict_fn = nlits_client.request_multiple_from_ids_triplets
    n_try = 10
    cursor = 0
    all_save_entries: List[SegmentationTrials] = []
    while cursor < len(data):
        data_slice = data[cursor: cursor+query_batch_size]
        save_entry: List[SegmentationTrials] = do_batch_request(data_slice, n_try, predict_fn)
        all_save_entries.extend(save_entry)
        cursor += query_batch_size
        ticker.tick()

    save_path = get_nlits_segmentation_trial_save_path(split)
    save_list_to_jsonl_w_fn(all_save_entries, save_path, SegmentationTrials.to_json)


def do_batch_request(item_list, n_try, predict_fn: Callable[[List[Tuple[List, List, List]]], List]):
    c_log.info("do_batch_request")
    pk2 = PromiseKeeper(predict_fn)
    si_list = []
    for item in item_list:
        prem, hypo, label = item
        ts_input_list: List[Tuple[List, List, List]] = []
        ts_input_info_list = []
        for _ in range(n_try):
            st, ed = get_random_split_location(hypo)
            hypo1, hypo2 = split_into_two(hypo, st, ed)
            ts_input = prem, hypo1, hypo2
            ts_input_list.append(ts_input)
            ts_input_info_list.append((st, ed))

        comparison_future = SegmentationTrialInputs(
            prem, hypo, label,
            [MyPromise(ts_input, pk2).future() for ts_input in ts_input_list],
            ts_input_info_list
        )
        si_list.append(comparison_future)
    pk2.do_duty()
    save_entry = list(map(SegmentationTrials.from_sti, si_list))
    return save_entry


class SegmentationTrialWorker(WorkerInterface):
    def __init__(self, n_item_per_job, output_dir):
        self.output_dir = output_dir
        split = "train"
        self.n_item_per_job = n_item_per_job
        self.data: List[Tuple[List[int], List[int], int]] = load_pickle_from(nli_tokenized_path(split))
        model_path = cpath.get_canonical_model_path2("nli_ts_run87_0", "model_12500")
        strategy = get_strategy2(use_tpu=False, tpu_name=None, force_use_gpu=True)

        def model_factory():
            model: tf.keras.models.Model = load_local_decision_model(model_path)
            return model

        self.inference_helper = InferenceHelper(model_factory, dataset_factory_600_3, strategy)
        self.encoder_helper = TS600_3_Encoder()

    def _predict(self, triplet_payload):
        payload = self.encoder_helper.combine_ts_triplets(triplet_payload)
        stacked_output = self.inference_helper.predict(payload)
        output = reslice_local_global_decisions(stacked_output)
        return output

    def work(self, job_id):
        random.seed(0)
        st = self.n_item_per_job * job_id
        ed = st + self.n_item_per_job
        data_slice = self.data[st:ed]
        n_try = 10
        save_entry: List[SegmentationTrials] = do_batch_request(data_slice, n_try, self._predict)
        save_path = os.path.join(self.output_dir, str(job_id))
        save_list_to_jsonl_w_fn(save_entry, save_path, SegmentationTrials.to_json)


def main():
    n_item = 400 * 1000
    n_item_per_job = 5000
    n_jobs = ceil_divide(n_item, n_item_per_job)

    def factory(output_dir):
        return SegmentationTrialWorker(n_item_per_job, output_dir)

    w_path = get_nlits_segmentation_trial_subjob_save_dir()
    job_runner = JobRunnerS(w_path, n_jobs, "nlits_trials", factory)
    job_runner.auto_runner()


if __name__ == "__main__":
    main()

