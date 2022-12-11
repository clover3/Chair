from typing import List, Dict


def i2b(l: List[int]) -> List[bool]:
    return [bool(k) for k in l]


def get_acc_prec_recall(pred: List[bool], gold: List[bool]) -> Dict:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p, g in zip(pred, gold):
        if g and p :
            tp += 1
        elif not g and p:
            fp += 1
        elif g and not p:
            fn += 1
        elif not g and not p:
            tn += 1
        else:
            assert False

    acc = (tp+tn) / (tp+tn+fp+fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1
    recall = tp / (tp + fn) if (tp+fn) > 0 else 1
    f1 = 2 * prec * recall / (prec + recall) if (prec * recall) > 0 else 0
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def get_acc_prec_recall_i(pred: List[int], gold: List[int]) -> Dict:
    def convert(labels):
        for l in labels:
            if l not in [0, 1]:
                print("WARNING there is label {}".format(l))
        return [bool(l) for l in labels]

    return get_acc_prec_recall(convert(pred), convert(gold))

