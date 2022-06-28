from data_generator.tokenizer_wo_tf import get_tokenizer


def main():
    q = "Where is SIGIR 2022?"
    d = "SIGIR 2022 will be held in Madrid"
    tokenizer = get_tokenizer()

    q_tokens = tokenizer.tokenize(q)
    d_tokens = tokenizer.tokenize(d)

    print(q_tokens)
    print(d_tokens)

    return NotImplemented


if __name__ == "__main__":
    main()