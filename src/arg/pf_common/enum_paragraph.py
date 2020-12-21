from typing import List, Iterable, Callable

from arg.pf_common.base import Paragraph
from data_generator.subword_translate import Subword
from datastore.interface import load
from datastore.table_names import TokenizedCluewebDoc
from galagos.types import SimpleRankedListEntry
from list_lib import lmap, flatten


def subword_tokenize_functor(tokenizer, word: str) -> List[Subword]:
    word = tokenizer.basic_tokenizer.clean_text(word)
    word = word.lower()
    word = tokenizer.basic_tokenizer.run_strip_accents(word)
    subwords = tokenizer.wordpiece_tokenizer.tokenize(word)
    return subwords


# return maximum index where number of subword tokens in subword_tokens[start:index] does not exist max_len
def move_cursor(subword_tokens: List[List[Subword]], start: int, max_len: int):
    cursor_ed = start
    num_subword = 0

    def can_add_subwords():
        if cursor_ed < len(subword_tokens):
            return num_subword + len(subword_tokens[cursor_ed]) <= max_len
        else:
            return False

    while can_add_subwords():
        num_subword += len(subword_tokens[cursor_ed])
        cursor_ed += 1

    return cursor_ed


def enum_paragraph(step_size, subword_len,
                   subword_tokenize: Callable[[str], List[Subword]], doc: SimpleRankedListEntry) -> Iterable[Paragraph]:
    # load tokens and BERT subword tokens
    tokens = load(TokenizedCluewebDoc, doc.doc_id)
    subword_tokens: List[List[Subword]] = lmap(subword_tokenize, tokens)
    cursor = 0

    while cursor < len(subword_tokens):
        cursor_ed = move_cursor(subword_tokens, cursor, subword_len)
        yield Paragraph(doc_id=doc.doc_id, doc_rank=doc.rank, doc_score=doc.score,
                        subword_tokens=list(flatten(subword_tokens[cursor:cursor_ed])),
                        tokens=tokens[cursor:cursor_ed])
        cursor += step_size


