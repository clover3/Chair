from cache import load_from_pickle


def main():
    obj = load_from_pickle("dev_claim_paras")

    for cid, docs in obj:
        if cid=='952':
            for doc in docs[:20]:
                print(" ".join(doc))
                print()


main()