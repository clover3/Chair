import unicodedata
from typing import NewType, List, Tuple

from data_generator.tokenizer_wo_tf import _is_whitespace, _is_control
from list_lib import lmap

Subword = NewType('Subword', str)


def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def normalize_pt(text):
    text = _clean_text(text)
    # .replace("”", "\"").replace("“", "\"") \
    r = text.replace("``", "\"").replace("''", "\"").replace("`", "'") \
        .replace("-lrb-", "(").replace("-rrb-", ")") \
        .replace("-lsb-", "[").replace("-rsb-", "]") \
        .replace("…", "...") \
        .replace("«", "\"")
    if "#" == r:
        r = "shop;"
    #    .replace("&", "&amp;")
    return _run_strip_accents(r)


def move_cursur2():
    NotImplemented


# Move subword_cursor (step_size), return resulting subword_cursor and corresponding word_cursor
def move_cursor(subword_cursor: int,
                subword_tokens: List[Subword],
                word_cursor: int,
                word_tokens: List[str],
                step_size: int,
                debug=False) -> Tuple[int, int]:
    next_subword_cursor = min(len(subword_tokens), subword_cursor + step_size)
    subword_tokens_to_pass = subword_tokens[subword_cursor:]

    lower_tokens = lmap(lambda x:x.lower(), word_tokens[word_cursor:])
    def dbgprint(*args):
        if debug:
            print(*args)
        pass

    # To compare, concatenate the model's tokens with space removed
    #
    sep_char = "#"
    def convert_index_out(unnorm_tokens, subword_tokens_to_pass, target_idx):
        dbgprint("-------")
        dbgprint("word tokens", unnorm_tokens)
        tokens = lmap(normalize_pt, unnorm_tokens)
        dbgprint("word tokens", tokens)
        dbgprint("subword_tokens_to_pass", subword_tokens_to_pass)
        dbgprint("target_idx", target_idx)
        dbgprint("subword_tokens", subword_tokens_to_pass)
        def last_subword_empty():
            if target_idx < len(subword_tokens_to_pass):
                return subword_tokens_to_pass[target_idx].replace("_", "").replace(" ", "") == ""
            else:
                return False

        if last_subword_empty():
            target_idx = target_idx - 1
            dbgprint("Replace target_idx to previous", subword_tokens_to_pass[target_idx])
        chars_to_pass = "".join(subword_tokens_to_pass[:target_idx])
        text_idx = 0
        # dbgprint("prev text", prev_text)
        # now we want to find a token from raw_sentence which appear after prev_text equivalent
        punct = ["\"", '\'', '`', '«']
        def char_equal(c1, c2):
            if c1 == c2 :
                return True
            if c1 in punct and c2 in punct:
                return True
            else:
                return False

        def update_text_idx(target_char, text_idx):
            while chars_to_pass[text_idx] in [sep_char, " "]:
                dbgprint(chars_to_pass[text_idx], text_idx)
                text_idx += 1
            dbgprint(63, target_char, chars_to_pass[text_idx])
            if char_equal(target_char, chars_to_pass[text_idx]):
                dbgprint(chars_to_pass[text_idx], text_idx)
                text_idx += 1
            elif chars_to_pass[text_idx] in punct and text_idx-1 >=0 and chars_to_pass[text_idx-1] in punct:
                dbgprint(chars_to_pass[text_idx], text_idx)
                text_idx += 1
                text_idx = update_text_idx(target_char, text_idx)
            return text_idx

        t_idx = 0
        dbgprint(chars_to_pass)
        def beginning_mismatch():
            if tokens and chars_to_pass:
                word_first_char = tokens[0][0]
                subword_first_char = chars_to_pass[0]
                if not char_equal(word_first_char, subword_first_char):
                    return True
            return False

        if beginning_mismatch():
            dbgprint("Adjusting start")
            t_idx = 1
            aligned = False
            dbgprint(tokens[t_idx])
            dbgprint("tokens=", tokens)
            dbgprint("subword_tokens_to_pass=", subword_tokens_to_pass)

            while not aligned:
                while chars_to_pass[text_idx] != tokens[t_idx][0]:
                    # print(chars_to_pass[text_idx], end=" ")
                    text_idx += 1
                # print()
                if chars_to_pass[text_idx:].startswith(tokens[t_idx]):
                    aligned = True
                else:
                    text_idx += 1

        try:
            while t_idx < len(tokens):
                token = tokens[t_idx]
                dbgprint(token)
                for c in token:
                    # Here, previous char should equal prev_text[text_idx]
                    text_idx = update_text_idx(c, text_idx)
                    # Here, c should equal prev_text[text_idx-1]
                    if not char_equal(c, chars_to_pass[text_idx-1]):
                        print("word_tokens=", word_tokens)
                        print("subword_tokens=", subword_tokens)
                        print("word_cursor=", word_cursor)
                        print("subword_cursor=", subword_cursor)
                        print("tokens=", tokens)
                        print("subword_tokens_to_pass=", subword_tokens_to_pass)
                        print(target_idx)
                        print(text_idx)
                        print(c, chars_to_pass[text_idx-1], ord(c), ord(chars_to_pass[text_idx-1]))
                    assert char_equal(c, chars_to_pass[text_idx-1])
                t_idx += 1
            dummy = chars_to_pass[text_idx]
        except IndexError:
            # now, text_idx >= len(prev_text)
            # tokens[:t_idx] is shorter or equal to chars_to_pass
            # tokens[:t_idx+1] is longer than chars_to_pass,
            #  in other word, tokens[t_idx-1] does not contain subword[target_idx]
            #  and tokens[t_idx] contain subword[target_idx],
            #  unless tokens[t_idx] is special tokens that are removed in BERT tokenization
            if t_idx < len(tokens):
                dbgprint("target_token", tokens[t_idx])
            dbgprint("t_idx", t_idx)
            return t_idx
        raise Exception


    def convert_index_out2(unnorm_tokens, subword_tokens_to_pass, target_idx):
        dbgprint("-------")
        dbgprint("word tokens", unnorm_tokens)
        tokens = lmap(normalize_pt, unnorm_tokens)
        dbgprint("word tokens", tokens)
        dbgprint("subword_tokens_to_pass", subword_tokens_to_pass)
        dbgprint("target_idx", target_idx)
        dbgprint("subword_tokens", subword_tokens_to_pass)

        # TODO
        word_idx = 0
        sw_idx = 0
        while True:
            NotImplemented
        return NotImplemented

    rel_word_token_idx = convert_index_out(lower_tokens,
                                           subword_tokens_to_pass,
                                           next_subword_cursor - subword_cursor)

    next_word_cursor = word_cursor + rel_word_token_idx

    return next_subword_cursor, next_word_cursor


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

