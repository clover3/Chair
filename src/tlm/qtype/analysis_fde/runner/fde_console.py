from tlm.qtype.analysis_fde.fde_module import get_fde_module


def main():
    fde = get_fde_module()
    while True:
        sent1 = input("Content span: ")
        sent2 = input("Document: ")
        ret = fde.get_promising(sent1, sent2)
        print((sent1, sent2))
        print(ret)


if __name__ == "__main__":
    main()
