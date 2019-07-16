import random
from cache import *
from data_generator import tokenizer_b as tokenization
import path

def translate_mask2token_level(sent_list, target_tokens, mask_indice, tokenizer):
    basic_tokens = list([tokenizer.basic_tokenizer.tokenize(s) for s in sent_list])
    sub_tokens_tree = []
    for sent_tokens in basic_tokens:
        sub_tokens_tree.append(list())
        for token in sent_tokens:
            r = tokenizer.wordpiece_tokenizer.tokenize(token)
            sub_tokens_tree[-1].append(r)

    # Iter[List[tokens]]

    sent_idx = 0
    token_idx = 0
    local_subword_idx = 0
    global_subword_idx = 0
    mask_indice.sort()
    word_mask_indice = []
    mask_indice_idx = 0

    while sent_idx < len(basic_tokens) and mask_indice_idx < len(mask_indice):
        st = sub_tokens_tree[sent_idx][token_idx][local_subword_idx]

        assert target_tokens[global_subword_idx] == st

        if global_subword_idx == mask_indice[mask_indice_idx]:
            word_mask_indice.append((sent_idx, token_idx))
            mask_indice_idx += 1


        local_subword_idx += 1
        global_subword_idx += 1

        if local_subword_idx == len(sub_tokens_tree[sent_idx][token_idx]):
            token_idx += 1
            local_subword_idx = 0

        if token_idx == len(basic_tokens[sent_idx]):
            sent_idx += 1
            token_idx = 0

    return basic_tokens, word_mask_indice



def retrieve_candidate(inst, tokenizer):
    # tokenize,
    target_tokens, sent_list, prev_tokens, next_tokens, mask_indice = inst
    basic_tokens, word_mask_indice = translate_mask2token_level(sent_list, target_tokens, mask_indice, tokenizer)

    # stemming

    NotImplemented








def main():
    rng = random.Random(0)
    vocab_file = os.path.join(path.data_path, "bert_voca.txt")
    tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)

    spr = StreamPickleReader("robust_problems_")

    while spr.has_next():
        inst = spr.get_item()
        retrieve_candidate(inst, tokenizer)


if __name__ == "__main__":
    main()
