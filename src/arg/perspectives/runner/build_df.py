from collections import Counter
from typing import List

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from arg.perspectives.split_helper import train_split
from cache import save_to_pickle
from list_lib import lmap, flatten
from misc_lib import average


def build_df():
    claims, val = train_split()
    gold = get_claim_perspective_id_dict()

    tokenizer = KrovetzNLTKTokenizer()
    df = Counter()

    dl_list = []
    for claim in claims:
        cid = claim["cId"]
        gold_pids = flatten(gold[cid])
        p_text_list: List[str] = lmap(perspective_getter, gold_pids)
        tokens_list = lmap(tokenizer.tokenize_stem, p_text_list)
        dl_list.extend(lmap(len, tokens_list))

        for t in set(flatten(tokens_list)):
            df[t] += 1

    print(dl_list)
    print("Avdl", average(dl_list))
    print(len(claims))
    print(df.most_common(30))
    save_to_pickle(df, "pc_df")


if __name__ == "__main__":
    build_df()

