from data_generator import tokenizer_wo_tf as tokenization


def take(v):
    return v.int64_list.value


class Feature2Text:
    def __init__(self, feature, tokenizer):
        self.tokenizer = tokenizer
        self.feature = feature

    def get_feature_by_need(self, feature_name):
        if not hasattr(self, feature_name):
            setattr(self, feature_name, take(self.feature[feature_name]))

        return getattr(self, feature_name)

    def get_input_ids(self):
        return self.get_feature_by_need("input_ids")

    def get_d_input_ids(self):
        return self.get_feature_by_need("d_input_ids")

    def get_masked_lm_ids(self):
        return self.get_feature_by_need("masked_lm_ids")

    def get_masked_lm_positions(self):
        return self.get_feature_by_need("masked_lm_positions")

    def get_selected_word(self):
        return self.get_feature_by_need("selected_word")

    def get_d_location_ids(self):
        return self.get_feature_by_need("d_location_ids")

    def get_input_tokens(self):
        return self.tokenizer.convert_ids_to_tokens(self.get_input_ids())

    def get_mask_answer_dict(self):
        mask_ans = {}
        masked_terms = self.tokenizer.convert_ids_to_tokens(self.get_masked_lm_ids())
        for pos, id in zip(list(self.get_masked_lm_positions()), masked_terms):
            mask_ans[pos] = id
        return mask_ans

    def get_selected_word_text(self):
        return self.ids_to_pretty_text(self.get_selected_word())

    def ids_to_pretty_text(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokenization.pretty_tokens(tokens, True)

    def get_input_as_text(self, resolve_mask=False, highlight_lookup_word=False):
        if resolve_mask:
            mask_ans = self.get_mask_answer_dict()

        if highlight_lookup_word:
            d_location_ids = self.get_d_location_ids()
            word = self.get_selected_word_text()
            emph_word = "<b>" + word + "</b>"

        tokens = self.tokenizer.convert_ids_to_tokens(self.get_input_ids())
        for i in range(len(tokens)):
            if resolve_mask and tokens[i] == "[MASK]":
                tokens[i] = "[MASK_{}: {}]".format(i, mask_ans[i])
            if highlight_lookup_word and i in d_location_ids and i != 0:
                print(i, emph_word)
                if tokens[i - 1] != emph_word:
                    tokens[i] = emph_word
                else:
                    tokens[i] = "-"

        return tokenization.pretty_tokens(tokens, True)

    def get_def_as_text(self):
        return self.ids_to_pretty_text(self.get_d_input_ids())