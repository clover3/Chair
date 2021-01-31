import random
from typing import List, Dict

from tlm.robust.load import load_robust_tokens_for_predict


def main():
    tokens_d: Dict[str, List[str]] = load_robust_tokens_for_predict(4)

    doc_ids = list(tokens_d.keys())
    random.shuffle(doc_ids)

    def enum_window(tokens, window):
        idx = 0
        while idx < len(tokens):
            cur_tokens = all_tokens[idx: idx + window]
            yield cur_tokens
            idx += window

    for doc_id in doc_ids[:10]:
        all_tokens = tokens_d[doc_id]
        print(doc_id)
        print("Fixed segmentation>")
        for tokens in enum_window(all_tokens, 128):
            print("< 128 tokens > ")
            print(" ".join(tokens))
            print("< 15 tokens > ")
            for cur_tokens in enum_window(tokens, 15):
                print(" ".join(cur_tokens))

        print()
        ##

if __name__ == "__main__":
    main()