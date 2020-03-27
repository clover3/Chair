import sys
from collections import Counter

from data_generator.argmining.eval import load_preditions, compare
from evals.tfrecord import load_tfrecord
from task.metrics import eval_3label


def do_eval(tfrecord_path, prediction_path):
    tfrecord = list(load_tfrecord(tfrecord_path))
    predictions = list(load_preditions(prediction_path))
    golds, preds = zip(*compare(tfrecord, predictions))
    golds = golds[:len(preds)]

    count = Counter()

    for label, pred in zip(golds, preds):
        count[(label, pred)] += 1

    print("\t0\t1\t2")
    for i in range(3):
        print("Gold {}".format(i), end="\t")
        for j in range(3):
            print(count[(i,j)], end="\t")
        print("")
    acc = (count[(0,0)] + count[(1,1)] +count[(2,2)]  ) / sum(count.values())
    print("Acc : ", acc)
    for i in range(3):
        prec = count[(i, i)] / sum([count[(j, i)] for j in range(3)])
        recall = count[(i, i)] / sum([count[(i, j)] for j in range(3)])
        f1 = 2 * prec * recall / (prec + recall)
        print("Label ", i)
        print("P/R/F1", prec, recall, f1)

    all_result = eval_3label(preds, golds)
    f1 = sum([result['f1'] for result in all_result]) / 3
    print("Macro Avg F1:", f1)

if __name__ == "__main__":
    print(do_eval(sys.argv[1], sys.argv[2]))