

def binary_kappa(annot1, annot2):
    agree_cnt = sum(list([a == b for a,b in zip(annot1, annot2)]))

    total = len(annot1)
    p0 = agree_cnt / total
    pe = (sum(annot1) / total) * (sum(annot2) / total) +   (1-(sum(annot1) / total)) * (1- (sum(annot2) / total))
    print(p0, pe)
    kappa = (p0 - pe) / (1 - pe)
    return kappa, p0


