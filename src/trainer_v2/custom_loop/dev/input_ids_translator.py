from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens


def main():

    tokenizer = get_tokenizer()
    input_ids_list =[
        [101, 1996, 2047, 2916, 2024, 3835, 2438, 102, ],
        [101,  1013,  1996, 14751,  6666,   102,],
        [101,  3071,  2428,  7777, 1013,   102,]
    ]

    for input_ids in input_ids_list:
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(pretty_tokens(tokens, True))


if __name__ == "__main__":
    main()

