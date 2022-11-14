from data_generator.tokenizer_wo_tf import FullTokenizerWarpper, _truncate_seq_pair


class EncoderUnitK:
    def __init__(self, max_sequence, voca_path):
        self.encoder = FullTokenizerWarpper(voca_path)
        self.max_seq = max_sequence
        self.CLS_ID = self.encoder.ft.convert_tokens_to_ids(["[CLS]"])[0]
        self.SEP_ID = self.encoder.ft.convert_tokens_to_ids(["[SEP]"])[0]

    def encode_token_pairs(self, tokens_a, tokens_b):
        ids_1 = self.encoder.ft.convert_tokens_to_ids(tokens_a)
        ids_2 = self.encoder.ft.convert_tokens_to_ids(tokens_b)
        return self.encode_from_ids(ids_1, ids_2)

    def encode_pair(self, text1, text2):
        ids_1 = self.encoder.encode(text1)
        ids_2 = self.encoder.encode(text2)
        return self.encode_from_ids(ids_1, ids_2)

    def encode_from_ids(self, id_tokens_a, id_tokens_b):
        _truncate_seq_pair(id_tokens_a, id_tokens_b, self.max_seq - 3)
        tokens = []
        segment_ids = []
        tokens.append(self.CLS_ID)
        segment_ids.append(0)
        for token in id_tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(self.SEP_ID)
        segment_ids.append(0)

        if id_tokens_b:
            for token in id_tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append(self.SEP_ID)
            segment_ids.append(1)

        input_ids = tokens

        while len(input_ids) < self.max_seq:
            input_ids.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq
        assert len(segment_ids) == self.max_seq
        return input_ids, segment_ids

