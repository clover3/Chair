import os
import random
from typing import List, Tuple
from cache import load_pickle_from
from data_generator.NLI.nli_info import nli_tokenized_path
from port_info import KERAS_NLI_PORT, LOCAL_DECISION_PORT
from trainer.promise import PromiseKeeper, MyPromise
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.per_task.cip.cip_common import get_random_split_location, split_into_two, Comparison, \
    ComparisonF, get_statistics
from trainer_v2.keras_server.bert_like_client import BERTClient
from trainer_v2.keras_server.nlits_client import NLITSClient


def get_statistics_from_direct_calc(
        nli_server_addr,
        nltis_server_addr,
        base_seq_length,
):
    random.seed(0)
    split = "train"
    data: List[Tuple[List[int], List[int], int]] = load_pickle_from(nli_tokenized_path(split))

    nli_client: BERTClient = BERTClient(nli_server_addr, KERAS_NLI_PORT, base_seq_length)
    nlits_client: NLITSClient = NLITSClient(nltis_server_addr, LOCAL_DECISION_PORT, base_seq_length)

    n_items = 5000
    n_try = 1
    iter_items = iter(data)

    pk1 = PromiseKeeper(nli_client.request_multiple_from_ids_pairs)
    pk2 = PromiseKeeper(nlits_client.request_multiple_from_ids_triplets)

    c_log.info("Init")
    cf_list = []
    for _ in range(n_items):
        item = next(iter_items)
        prem, hypo, label = item
        full_input = prem, hypo
        ts_input_list: List[Tuple[List, List, List]] = []
        ts_input_info_list = []
        for _ in range(n_try):
            st, ed = get_random_split_location(hypo)
            hypo1, hypo2 = split_into_two(hypo, st, ed)
            ts_input = prem, hypo1, hypo2
            ts_input_list.append(ts_input)
            ts_input_info_list.append((st, ed))

        comparison_future = ComparisonF(
            prem, hypo, label,
            MyPromise(full_input, pk1).future(),
            [MyPromise(ts_input, pk2).future() for ts_input in ts_input_list],
            ts_input_info_list
        )
        cf_list.append(comparison_future)

    c_log.info("Sending payloads")
    pk1.do_duty()
    pk2.do_duty()
    c_log.info("Done")

    def iterate_comparison():
        for i in range(n_items):
            cf = cf_list[i]
            comparison = Comparison.from_comparison_f(cf)
            yield comparison

    get_statistics(iterate_comparison())


def main():
    get_statistics_from_direct_calc(os.environ['NLI_ADDR'], os.environ['NLITS_ADDR'], 300)


if __name__ == "__main__":
    main()