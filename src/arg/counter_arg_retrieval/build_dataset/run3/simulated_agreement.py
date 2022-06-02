from evals.agreement import binary_kappa


def main():
    paired = []
    paired.extend([(0, 0)] * 26)
    paired.extend([(0, 1)] * 16)
    paired.extend([(1, 0)] * 2)
    paired.extend([(1, 1)] * 10)
    annot1, annot2 = zip(*paired)
    kappa, p0 = binary_kappa(annot1, annot2)
    print(kappa, p0)


if __name__ == "__main__":
    main()