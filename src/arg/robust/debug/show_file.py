from tlm.robust.load import load_robust_tokens_for_train


def main():
    tokens_d = load_robust_tokens_for_train()
    print("Tokens loaded")
    while True:
        try:
            doc_id = input()
            print("doc_id: ", doc_id)
            print(tokens_d[doc_id])
            print("----")
            ##
        except KeyError as e:
            print("Doc not found")
            print(e)
        except Exception:
            pass



if __name__ == "__main__":
    main()