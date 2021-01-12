from collections import Counter
from typing import List, Tuple

from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs
from arg.qck.decl import QCKQuery, KDP, QKUnit
from arg.qck.instance_generator.qknc_datagen import QKInstanceGenerator
# Make payload without any annotation
from cache import load_from_pickle
from exec_lib import run_func_with_config


def main(config):
    qk_candidate_name = config['qk_candidate_name']
    gold_qks: List[QKUnit] = load_from_pickle(config['qk_candidate_gold'])

    def get_qk_key_str(query: QCKQuery, kdp: KDP) -> Tuple:
        key = query.query_id, kdp.to_str()
        return key

    gold_keys = set()
    for q, k_list in gold_qks:
        for k in k_list:
            key = get_qk_key_str(q, k)
            gold_keys.add(key)

    counter = Counter()

    def is_correct(query: QCKQuery, passage: KDP) -> int:
        key = get_qk_key_str(query, passage)
        if key in gold_keys:
            counter[1] += 1
            return 1
        else:
            counter[0] += 1
            return 0

    start_generate_jobs(QKInstanceGenerator(is_correct),
                        config['split'],
                        qk_candidate_name,
                        config['name_prefix'])

    print("qk label distribution")
    print(counter)


if __name__ == "__main__":
    run_func_with_config(main)