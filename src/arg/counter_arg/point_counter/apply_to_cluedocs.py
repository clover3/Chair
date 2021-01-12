from typing import List

from sklearn.svm import LinearSVC

from arg.counter_arg.runner_qck.qck_datagen import load_qk
from arg.qck.decl import QKUnit, KDP
from cache import load_from_pickle


def main():
    split = "training"
    qk_list: List[QKUnit] = load_qk(split)
    svclassifier: LinearSVC = load_from_pickle("svclassifier")
    feature_extractor = load_from_pickle("feature_extractor")

    def get_score(k: KDP) -> float:
        text = " ".join(k.tokens)
        x = feature_extractor.transform([text])
        s = svclassifier._predict_proba_lr(x)
        return s[0][0]

    for q, k_list in qk_list:
        print("Query:", q.text)
        for kdp in k_list:
            score = get_score(kdp)
            if score > 0.8:
                print(score, " ".join(kdp.tokens))


if __name__ == "__main__":
    main()