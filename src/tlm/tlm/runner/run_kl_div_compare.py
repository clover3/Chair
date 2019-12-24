import os

from cie.arg import kl
from tlm.unigram_lm_from_tfrecord import get_lm_tf


def compare_ukp_and_wiki():
    ukp_a = "/mnt/nfs/work3/youngwookim/data/stance_small_tf/tf_enc_a"
    ukp_b = "/mnt/nfs/work3/youngwookim/data/stance_small_tf/tf_enc_b"
    wiki_c = "/mnt/nfs/work3/youngwookim/data/bert_tf/unmasked_pair_x3/1"
    lm_a = get_lm_tf(ukp_a, 100, True)
    lm_b = get_lm_tf(ukp_b, 100, True)
    lm_c = get_lm_tf(wiki_c, 100, True)

    div = kl.kl_divergence(lm_a, lm_b)
    print("KL Divergence a,b", div)
    div = kl.kl_divergence(lm_a, lm_c)
    print("KL Divergence a,c", div)


def compare_nli_and_wiki():
    from path import output_path
    nli_dev_fiction = os.path.join(output_path, "nli_tfrecord_per_genre", "dev_fiction")
    nli_train_fiction = os.path.join(output_path, "nli_tfrecord_per_genre", "train_fiction")
    nli_dev_telephone = os.path.join(output_path, "nli_tfrecord_per_genre", "dev_telephone")
    wiki_c = os.path.join(output_path, 'unmasked_pair_x3_0')
    lm_dev_fiction = get_lm_tf(nli_dev_fiction, 100, True)
    lm_train_fiction = get_lm_tf(nli_train_fiction, 100, True)
    lm_dev_telephone = get_lm_tf(nli_dev_telephone, 100, True)
    lm_wiki_c = get_lm_tf(wiki_c, 100, True)

    print("KL Divergence fiction train vs dev", kl.kl_divergence(lm_dev_fiction, lm_train_fiction))
    print("KL Divergence dev fiction vs telephone", kl.kl_divergence(lm_dev_fiction, lm_dev_telephone))
    print("KL Divergence dev fiction vs wiki", kl.kl_divergence(lm_dev_fiction, lm_wiki_c))


if __name__ == "__main__":
    compare_nli_and_wiki()