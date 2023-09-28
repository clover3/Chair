import sys

from cpath import output_path
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sub_samples_as_qd_pair
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import eval_on_train_when_0, \
    predict_and_save_scores_w_itr
from trainer_v2.per_project.transparency.mmp.when_corpus_based.feature_encoder import BM25TFeatureEncoder
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import get_mmp_bm25, get_candidate_voca
from trainer_v2.per_project.transparency.transformers_utils import pad_truncate
import tensorflow as tf



class Scorer:
    def __init__(self, scoring_layer):
        bm25 = get_mmp_bm25()
        self.candidate_voca = get_candidate_voca()
        # self.voca_size = len(candidate_voca) + 1
        # print("voca_size", self.voca_size)
        self.voca_size = 699 + 1
        feature_encoder = BM25TFeatureEncoder(bm25, self.candidate_voca)
        self.process_fn = feature_encoder.get_term_translation_weight_feature
        self.scoring_layer = scoring_layer

    def score(self, q, d):
        score, feature_ids, feature_values = self.process_fn(q, d)
        feature_ids = pad_truncate(feature_ids, 32)
        feature_values = pad_truncate(feature_values, 32)
        x = tf.scatter_nd(tf.expand_dims(feature_ids, 1),
                          feature_values, [self.voca_size])
        logits = self.scoring_layer(tf.expand_dims(x, 0))[0]
        return score + logits


def get_scorer(model_name):
    model_path = path_join(
        output_path, "model", "runs", model_name)
    model = tf.keras.models.load_model(model_path)
    maybe_scorer = model.layers[3]
    scorer = Scorer(maybe_scorer)
    return scorer


def main():
    model_name = sys.argv[1]
    scorer = get_scorer(model_name)
    run_name = model_name
    dataset = "train_when_0"
    n_item = 230958
    itr = load_msmarco_sub_samples_as_qd_pair(dataset)
    predict_and_save_scores_w_itr(scorer.score, dataset, run_name, itr, n_item)
    score = eval_on_train_when_0(run_name)
    print(f"MRR:\t{score}")



if __name__ == "__main__":
    main()