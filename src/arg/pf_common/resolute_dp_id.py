import os
import pickle
from functools import partial
from typing import List, Dict, Tuple

from arg.perspectives.cpid_def import CPID
from arg.pf_common.base import ParagraphFeature, DPID
from data_generator.common import get_tokenizer
from data_generator.tokenizer_wo_tf import FullTokenizer
from list_lib import lmap


def get_cpids_and_token_keys(tokenizer: FullTokenizer,
                             para_feature: ParagraphFeature) -> Tuple[str, DPID]:
    text1 = para_feature.datapoint.text1
    tokens1 = tokenizer.tokenize(text1)
    text2 = para_feature.datapoint.text2
    tokens2 = tokenizer.tokenize(text2)

    key = " ".join(tokens1[1:]) + "_" + " ".join(tokens2)
    dpid: DPID = para_feature.datapoint.id
    return key, dpid


def pickle_resolute_dict(input_dir, st, ed):
    tokenizer = get_tokenizer()
    tokens_to_dpid = {}
    for job_id in range(st, ed):
        features: List[ParagraphFeature] = pickle.load(open(os.path.join(input_dir, str(job_id)), "rb"))
        f = partial(get_cpids_and_token_keys, tokenizer)

        d: Dict[str, CPID] = dict(lmap(f, features))
        tokens_to_dpid.update(d)

    pickle.dump(tokens_to_dpid, open("resolute_dict_{}_{}".format(st, ed), "wb"))
    print("Done")