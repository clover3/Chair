import sys
import time
import tensorflow as tf

from misc_lib import TimeEstimator
from trainer_v2.per_project.transparency.splade_regression.iterate_data import iterate_triplet
from trainer_v2.per_project.transparency.splade_regression.splade_predictor import get_splade
from trainer_v2.train_util.get_tpu_strategy import get_strategy2


def predict_single(payload, splade):
    output = []
    for text in payload:
        v = splade.encode(text)
        output.append(v)


def predict_by_batch(payload, splade):
    batch_size = 16
    output = []

    cursor = 0
    while cursor < len(payload):
        v_numpy = splade.encode_batch(payload[cursor: cursor + batch_size])
        cursor += batch_size
        for i in range(len(v_numpy)):
            output.append(v_numpy[i])


def build_payload():
    payload = []
    cnt = 0
    for item in iterate_triplet(sys.argv[1]):
        query, doc1, doc2 = item
        payload.append(query)
        payload.append(doc1)
        payload.append(doc2)
        cnt += 1
        if cnt == 10:
            break
    return payload


def do_measure(predict_fn, splade):
    dummy = splade.encode("dummy_text")
    payload = build_payload()
    st = time.time()
    predict_fn(payload, splade)
    ed = time.time()
    print("{} per 10 triplets".format(ed-st))


def main():
    # strategy = get_strategy2(False)
    # with strategy.scope():
    splade = get_splade()
    # do_measure(predict_single, splade)
    do_measure(predict_by_batch, splade)


if __name__ == "__main__":
    main()