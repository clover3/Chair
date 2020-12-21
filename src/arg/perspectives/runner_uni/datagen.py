import os
from typing import List, Dict

from arg.perspectives.claim_lm.token_score_datagen import get_generator, Record, write_records
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms
from base_type import FilePath
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from list_lib import lmap
from misc_lib import exist_or_mkdir
from models.classic.lm_util import average_counters

env_data_dir = os.environ["DATA_DIR"]


def main():
    d_ids = list(load_train_claim_ids())
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    save_name = "pc_token_train"
    return do_datagen(d_ids, q_res_path, save_name)


def do_datagen(d_ids, q_res_path, save_name):
    claims: List[Dict] = get_claims_from_ids(d_ids)
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    claim_lms = build_gold_lms(claims)
    bg_lm = average_counters(lmap(lambda x: x.LM, claim_lms))
    alpha = 0.1
    max_seq_length = 512
    generator = get_generator(max_seq_length, bg_lm, alpha)
    out_dir = os.path.join(env_data_dir, save_name)
    exist_or_mkdir(out_dir)
    for claim_lm in claim_lms:
        print(claim_lm.cid)
        records: List[Record] = generator(claim_lm, ranked_list[str(claim_lm.cid)])
        output_path = os.path.join(out_dir, str(claim_lm.cid))
        write_records(records, max_seq_length, output_path)


if __name__ == "__main__":
    main()
