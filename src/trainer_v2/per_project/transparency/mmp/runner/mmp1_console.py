from cpath import pjoin, data_path, get_canonical_model_path
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    voca_path = pjoin(data_path, "bert_voca.txt")
    batch_size = 32
    model_path = get_canonical_model_path("mmp1")
    strategy = get_strategy()
    with strategy.scope():
        score_fn = get_scorer(model_path, batch_size)

    while True:
        query = input("Query: ")
        doc = input("Document: ")

        score = score_fn([(query, doc)])[0]
        print("score: ", score)



if __name__ == "__main__":
    main()
