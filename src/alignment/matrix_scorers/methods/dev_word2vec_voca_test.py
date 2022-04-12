from collections import Counter
from typing import List

import gensim

from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from alignment.matrix_scorers.methods.get_word2vec_scorer import get_word2vec_path
from alignment.nli_align_path_helper import load_mnli_rei_problem
from data_generator.tokenizer_wo_tf import ids_to_text, get_tokenizer


def enum_words():
    dataset = "dev"
    problems: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset)
    tokenizer = get_tokenizer()

    for p in problems:
        for seg_text in [p.seg_instance.text1, p.seg_instance.text2]:
            for seg_idx in seg_text.enum_seg_idx():
                ids = seg_text.get_tokens_for_seg(seg_idx)
                word = ids_to_text(tokenizer, ids)
                yield word

def main():
    print("Reading word2vec")
    w2v = gensim.models.KeyedVectors.load_word2vec_format(get_word2vec_path(), binary=True)
    print("Testing voca")
    counter = Counter()
    seen = set()
    for word in enum_words():
        if word in seen:
            pass
        else:
            seen.add(word)
            if word in w2v:
                counter['match'] += 1
            else:
                if word.capitalize() in w2v:
                    counter["capitalize"] += 1
                elif word.upper() in w2v:
                    counter["upper"] += 1
                elif len(word) == 1:
                    counter["Single char"] += 1
                else:
                    counter["not found"] += 1
                    print(word)

    print(counter)

if __name__ == "__main__":
    main()