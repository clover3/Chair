from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from typing import List, Iterable, Callable, Dict, Tuple, Set

from alignment.matrix_scorers.methods.get_word2vec_scorer import get_word2vec_path
from alignment.nli_align_path_helper import load_mnli_rei_problem
from data_generator.tokenizer_wo_tf import ids_to_text, get_tokenizer
import gensim

def enum_words():
    dataset = "train"
    problems: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset)
    tokenizer = get_tokenizer()

    for p in problems:
        for seg_text in [p.seg_instance.text1, p.seg_instance.text2]:
            for seg_idx in seg_text.enum_seg_idx():
                ids = seg_text.get_tokens_for_seg(seg_idx)
                word = ids_to_text(ids, tokenizer)
                yield word

def main():
    w2v = gensim.models.KeyedVectors.load_word2vec_format(get_word2vec_path(), binary=True)

    for word in enum_words():
        if word not in w2v:
            print(word)


if __name__ == "__main__":
    main()