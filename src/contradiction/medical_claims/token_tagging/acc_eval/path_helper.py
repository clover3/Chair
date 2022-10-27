import os
from typing import List

from contradiction.medical_claims.token_tagging.path_helper import get_sbl_binary_label_path
from contradiction.token_tagging.acc_eval.label_loaders import SentTokenLabel, load_sent_token_label
from misc_lib import warn_value_one_of


def load_sbl_binary_label(tag, split) -> List[SentTokenLabel]:
    warn_value_one_of(split, ["val", "test"])
    file_path = get_sbl_binary_label_path(tag, split)
    return load_sent_token_label(file_path)

