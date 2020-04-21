from collections import Counter

from data_generator.common import get_tokenizer
from misc_lib import get_dir_files, TimeEstimator
from tf_util.enum_features import load_record_v2
from tlm.data_gen.feature_to_text import take


def count_terms(file_path):
    counter = Counter()

    for feature in load_record_v2(file_path):
        input_ids = take(feature["input_ids"])
        alt_emb_mask = take(feature["alt_emb_mask"])

        cur_words = []
        for i in range(len(input_ids)):
            if alt_emb_mask[i] :
                cur_words.append(input_ids[i])
            else:
                if cur_words:
                    sig = " ".join([str(num) for num in cur_words])
                    counter[sig] += 1
                cur_words = []
    return counter


def count_terms_for_dir(dir_path):
    def sig_to_terms(sig: str):
        token_ids = [int(t) for t in sig.split(" ")]
        terms = tokenizer.convert_ids_to_tokens(token_ids)
        return "".join(terms)

    counter = Counter()
    file_list = get_dir_files(dir_path)
    ticker = TimeEstimator(len(file_list))
    for file_path in file_list:
        counter.update(count_terms(file_path))
        ticker.tick()

    tokenizer = get_tokenizer()

    for sig, cnt in counter.items():
        term = sig_to_terms(sig)
        print(term, cnt)

    return


def count_new_term():
    dir_path = "/mnt/nfs/work3/youngwookim/data/bert_tf/nli_dev_new_voca"
    count_terms_for_dir(dir_path)


if __name__ == "__main__":
    count_new_term()