from typing import Dict

from iter_util import load_jsonl
from misc_lib import NamedAverager
from trainer_v2.epr.path_helper import get_alignment_path


def main():
    job_id = 0

    entries = load_jsonl(get_alignment_path("snli", "train", str(job_id)))
    idx_key_list = ["max_h_idx_for_p", "max_p_idx_for_h"]
    idx_keys_idx: Dict[str, int] = {
        "premise": 0,
        "hypothesis": 1
     }
    sent_type_list = ["premise", "hypothesis"]
    avger = NamedAverager()
    for entry in entries:
        for sent_type in sent_type_list:
            num_segs = len(entry[sent_type])
            key_idx: int = idx_keys_idx[sent_type]
            max_idx_cur = entry[idx_key_list[key_idx]]
            max_idx_opposite = entry[idx_key_list[1-key_idx]]

            cur_sent = entry[sent_type]
            other_sent = entry[sent_type_list[1-key_idx]]
            accepted_align = 0
            exact_match = 0
            for seg_idx in range(num_segs):
                j = max_idx_cur[seg_idx]
                if max_idx_opposite[j] == seg_idx:
                    accepted_align += 1

                if other_sent:
                    cur_token = cur_sent[seg_idx]
                    other_token = other_sent[j]
                    if cur_token.lower() == other_token.lower():
                        exact_match += 1
            try:
                accepted_rate = accepted_align / num_segs
                exact_match_rate = exact_match / num_segs
                avger[sent_type].append(accepted_rate)
                avger[sent_type + "_exact"].append(exact_match_rate)
            except ZeroDivisionError:
                pass

    for key, value in avger.get_average_dict().items():
        print(key, value)


if __name__ == "__main__":
    main()
