from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01


def main():
    bm25 = get_bm25_mmp_25_01_01()

    my_k1 = bm25.core.k1
    avdl = bm25.core.avdl
    b = bm25.core.b

    idf = bm25.term_idf_factor("when")
    dl = 2
    print("f, second")
    for f in [0, 1, 2, 5, 10, 20]:
        K = my_k1 * ((1-b) + b * (float(dl)/float(avdl)))
        second = ((my_k1 + 1) * f) / (K + f)

        print(f, second)





if __name__ == "__main__":
    main()