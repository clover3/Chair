import numpy as np

from base_type import FileName
from cpath import pjoin, output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc.show_checkpoint_vars import load_checkpoint_vars


def compare_before_after():
    tokenizer = get_tokenizer()

    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("heavy metal"))
    dir_name = pjoin(pjoin(output_path, FileName("model")), FileName("alt_emb_heavy_metal_D"))
    before = pjoin(dir_name, FileName("model.ckpt-0"))
    after = pjoin(dir_name, FileName("model.ckpt-10000"))

    v1_d = load_checkpoint_vars(before)
    v2_d = load_checkpoint_vars(after)

    for key in v1_d :
        if key in v2_d:
            s = np.sum(v1_d[key] - v2_d[key])
            if np.abs(s) > 0.01:
                print(key, s)

    ori_emb = v2_d['bert/embeddings/word_embeddings']
    alt_emb_before = v1_d['bert/embeddings/word_embeddings_alt']
    alt_emb_after = v2_d['bert/embeddings/word_embeddings_alt']


    def show_diff_from_ori(token_id):
        diff = np.sum(np.abs(ori_emb[token_id] - alt_emb_after[token_id]))
        print(token_id, diff)

    def show_diff_from_step0(token_id):
        diff = np.sum(np.abs(alt_emb_before[token_id] - alt_emb_after[token_id]))
        print(token_id, diff)


    print("Diff against original embedding")
    print("Target words")
    for token_id in ids:
        show_diff_from_ori(token_id)

    print("Random words")
    for token_id in [321, 598, 5854]:
        show_diff_from_ori(token_id)

    print("Diff against step0 random init embedding")
    print("Target words")
    for token_id in range(0, 30000):
        diff = np.sum(np.abs(alt_emb_before[token_id] - alt_emb_after[token_id]))
        if diff > 0.001:
            print(token_id, diff)



if __name__ == "__main__":
    compare_before_after()