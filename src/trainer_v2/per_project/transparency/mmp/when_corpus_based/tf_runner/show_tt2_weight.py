import sys

import tensorflow as tf


from cpath import output_path
from misc_lib import path_join

from trainer_v2.per_project.transparency.mmp.when_corpus_based.tf_runner.rerank_with_tt2 import get_scorer


def main():
    model_name = sys.argv[1]
    print("Using ", model_name)
    scorer = get_scorer(model_name)
    model_path = path_join(
        output_path, "model", "runs", model_name)
    model = tf.keras.models.load_model(model_path)
    maybe_scorer = model.layers[3]

    maybe_scorer = model.layers[3]
    w = maybe_scorer.weights[0]
    print(w.shape)
    idx_to_voca = {v: k for k, v in scorer.candidate_voca.items()}
    rank = tf.argsort(w)[::-1]
    print(rank)

    when_id = scorer.candidate_voca['when']
    print("", "when", w[when_id].numpy())
    for i_tensor in rank[:200]:
        i = int(i_tensor)
        word = idx_to_voca[i]
        print(i, word, w[i].numpy())
    print()
    for i_tensor in rank[-20:]:
        i = int(i_tensor)
        try:
            word = idx_to_voca[i]
        except KeyError:
            word= "-"
        print(i, word, w[i].numpy())
    # print(maybe_scorer.weights)
    # ret = scorer.score("when is today", "today is sunday")
    # print(ret)



if __name__ == "__main__":
    main()