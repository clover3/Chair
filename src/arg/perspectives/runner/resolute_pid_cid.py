import os
import pickle
import sys
from functools import partial
from typing import List, Dict, Tuple

from arg.perspectives.cpid_def import CPID
from arg.perspectives.select_paragraph_perspective import ParagraphClaimPersFeature
from data_generator.tokenizer_wo_tf import FullTokenizer, get_tokenizer
from list_lib import lmap


def get_cpids_and_token_keys(tokenizer: FullTokenizer,
                             claim_entry: ParagraphClaimPersFeature) -> Tuple[str, CPID]:
    claim_text = claim_entry.claim_pers.claim_text
    claim_tokens = tokenizer.tokenize(claim_text)
    p_text = claim_entry.claim_pers.p_text
    p_tokens = tokenizer.tokenize(p_text)
    key = " ".join(claim_tokens) + "_" + " ".join(p_tokens)
    cpid: CPID = CPID("{}_{}".format(claim_entry.claim_pers.cid, claim_entry.claim_pers.pid))
    return key, cpid


def pickle_resolute_dict(input_dir, st, ed):
    tokenizer = get_tokenizer()
    tokens_to_cpid = {}
    for job_id in range(st, ed):
        print(job_id)
        features: List[ParagraphClaimPersFeature] = pickle.load(open(os.path.join(input_dir, str(job_id)), "rb"))
        f = partial(get_cpids_and_token_keys, tokenizer)

        d: Dict[str, CPID] = dict(lmap(f, features))
        tokens_to_cpid.update(d)

    pickle.dump(tokens_to_cpid, open("resolute_dict_{}_{}".format(st, ed), "wb"))
    print("Done")


# resolute_pid_cid.py
if __name__ == "__main__":
    pickle_resolute_dict(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
