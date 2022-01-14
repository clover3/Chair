from data_generator.tokenizer_wo_tf import EncoderUnitOld


class EncoderForNLI(EncoderUnitOld):
    def __init__(self, max_sequence, voca_path):
        super(EncoderForNLI, self).__init__(max_sequence, voca_path)
        CLS_ID = self.encoder.ft.convert_tokens_to_ids(["[CLS]"])[0]
        SEP_ID = self.encoder.ft.convert_tokens_to_ids(["[SEP]"])[0]

        self.CLS_ID = CLS_ID
        self.SEP_ID = SEP_ID

    def encode_token_pairs(self, maybe_query, maybe_document):
        tokens_a = maybe_document
        tokens_b = maybe_query
        ids_1 = self.encoder.ft.convert_tokens_to_ids(tokens_a)
        ids_2 = self.encoder.ft.convert_tokens_to_ids(tokens_b)
        d = self.encode_inner(ids_1, ids_2)
        return d["input_ids"], d["input_mask"], d["segment_ids"]