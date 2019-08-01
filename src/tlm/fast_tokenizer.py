from data_generator import tokenizer_b as tokenization

import path
import os

class FTokenizer:
    def __init__(self):
        self.d_num_sub_tokens = dict()
        vocab_file = os.path.join(path.data_path, "bert_voca.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=True)


    def num_sub_tokens(self, space_tokenized_token):
        if space_tokenized_token in self.d_num_sub_tokens:
            return self.d_num_sub_tokens[space_tokenized_token]

        t = space_tokenized_token.lower()
        t = self.tokenizer.basic_tokenizer._run_strip_accents(t)
        basic_tokens = self.tokenizer.basic_tokenizer._run_split_on_punc(t)

        l_all = 0
        for bt in basic_tokens:
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(bt)
            l_all += len(sub_tokens)

        self.d_num_sub_tokens[space_tokenized_token] = l_all
        return l_all

    def get_subtoken_loc(self, text, sub_loc_list):
        space_tokenized_tokens = tokenization.whitespace_tokenize(text)

        loc = 0
        sl_idx = 0
        sb_loc = 0
        corr_loc_list = []
        while loc < len(space_tokenized_tokens) and sl_idx < len(sub_loc_list):
            if sub_loc_list[sl_idx] == sb_loc:
                corr_loc_list.append(loc)
                sl_idx += 1
            elif sub_loc_list[sl_idx] < sb_loc:
                corr_loc_list.append(loc-1)
                sl_idx += 1

            t = space_tokenized_tokens[loc]
            loc += 1
            sb_loc += self.num_sub_tokens(t)

        return corr_loc_list

    def get_num_sbtokens_in_text(self, text):
        n = 0
        for t in tokenization.whitespace_tokenize(text):
            n += self.num_sub_tokens(t)
        return n

    def smart_cut(self, title, text, max_window):
        space_tokenized_tokens = tokenization.whitespace_tokenize(text)

        min_skip = int(max_window/2)

        title_sb_len = self.get_num_sbtokens_in_text(title)

        def get_segment_from(st_loc, st_sub_loc):
            ed_loc = st_loc
            ed_sub_loc = st_sub_loc

            t = space_tokenized_tokens[ed_loc]
            next_sub_len = self.num_sub_tokens(t)
            while title_sb_len + ed_sub_loc - st_sub_loc + next_sub_len < max_window:
                ed_loc += 1
                ed_sub_loc += next_sub_len
                if ed_loc >= len(space_tokenized_tokens):
                    break
                t = space_tokenized_tokens[ed_loc]
                next_sub_len = self.num_sub_tokens(t)

            assert title_sb_len + ed_sub_loc - st_sub_loc < max_window

            segment = title + "\n" + " ".join(space_tokenized_tokens[st_loc:ed_loc])
            assert type(segment) == type("hello")
            return segment, ed_loc

        def move_to_next_dot(st_loc, st_sub_loc):
            new_st_sub_loc = st_sub_loc
            new_st_loc = st_loc
            while new_st_sub_loc - st_sub_loc < min_skip\
                    and new_st_loc + 1 < len(space_tokenized_tokens):
                t = space_tokenized_tokens[new_st_loc]
                next_sub_len = self.num_sub_tokens(t)
                new_st_loc += 1
                new_st_sub_loc += next_sub_len

            t = space_tokenized_tokens[new_st_loc]
            next_sub_len = self.num_sub_tokens(t)

            f_dot_found = False
            # move until meeting dot '.' or max_window
            while not f_dot_found and \
                    new_st_sub_loc - st_sub_loc + next_sub_len < max_window and \
                    new_st_loc+1 < len(space_tokenized_tokens):

                if t == ".":
                    f_dot_found = True

                new_st_loc += 1
                t = space_tokenized_tokens[new_st_loc]
                new_st_sub_loc += next_sub_len

            return new_st_loc, new_st_sub_loc

        res = []
        st_loc = 0
        st_sub_loc = 0
        while st_loc + 4 < len(space_tokenized_tokens):
            # update ed_loc so that (ed_sub_loc-st_sub_loc) < window

            segment, ed_loc = get_segment_from(st_loc, st_sub_loc)
            # update LOC
            res.append((segment, st_loc, ed_loc))

            st_loc, st_sub_loc = move_to_next_dot(st_loc, st_sub_loc)
        return res