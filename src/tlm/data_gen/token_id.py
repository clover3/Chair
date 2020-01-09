import os

from cpath import data_path
from data_generator import tokenizer_wo_tf as tokenization


def pritn_token_id():
    vocab_file = os.path.join(data_path, "bert_voca.txt")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    tokens = ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    for token, id in zip(tokens, ids):
        print(token, id)




if __name__ == "__main__":
    pritn_token_id()