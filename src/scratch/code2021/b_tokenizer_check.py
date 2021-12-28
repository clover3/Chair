from data_generator.tokenizer_wo_tf import get_tokenizer


def main():
    tokenizer = get_tokenizer()

    wildcard = [
                "chihuahua",
                "malacca",
                "vimes",
                "donnie",
                "plaques",
                "bangs",
                "floppy",
                "huntsville",
                "loretta",
                "nikolay",
                ]

    text = "hello world [MASK] how"

    for w in wildcard:
        if w not in text:
            break_w = w

    new_text = text.replace("[MASK]", break_w)
    tokens = tokenizer.tokenize(new_text)
    tokens = [t if t!= break_w else "[MASK]" for t in tokens]
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    return NotImplemented


if __name__ == "__main__":
    main()