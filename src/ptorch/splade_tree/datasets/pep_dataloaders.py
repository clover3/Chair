
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from ..utils.utils import rename_keys
from transformers.tokenization_utils_base import BatchEncoding, EncodingFast


class PartitionEncoder:
    def __init__(self):
        pass

    def partition_query(self, query: EncodingFast) -> list[int]:

        pass

    def partition_doc(self, doc: EncodingFast) -> list[int]:
        pass


# encoded_pos has sequence like
# Query is partitioned into two segments and concatenated
# Seg_Query1 = [CLS] [Q_Token1] [Q_Token2] [SEP]
# Seg_Query2 = [CLS] [Q_Token3] ... [Q_Token_N] [SEP]
# DocSeg = [SEP] [D_Tokens1] [D_Tokens2] [D_Tokens3] [SEP] [D_Tokens4] [D_Tokens5] [SEP] [D_Tokens6] [SEP]
# Seg1 = Seg_Query1 + DocSeg
# Seg2 = Seg_Query2 + DocSeg
class PEPPairsDataLoader(DataLoaderWrapper):
    """Siamese encoding (query and document independent)
    train mode (pairs)
    """
    def __init__(self, tokenizer_type, max_length, **kwargs):
        super(PEPPairsDataLoader, self).__init__(tokenizer_type, max_length, **kwargs)
        self.p_encoder: PartitionEncoder = NotImplemented

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 3 (text) items (q, d_pos, d_neg)
        """
        q, d_pos, d_neg, s_pos, s_neg = zip(*batch)
        combine = []

        q_rep = self.tokenizer(q)

        indice_q_seg: list[list[int]] = [self.p_encoder.partition_query(q_rep[i]) for i in range(len(q))]

        d_pos_rep = self.tokenizer(d_pos)
        d_neg_rep = self.tokenizer(d_neg)

        partitioned_q_list = self.process(list(q))

        pos_entries = []
        neg_entries = []
        for i in range(len(q)):
            indice_q_seg = self.p_encoder.partition_query(q_rep[i])
            indices_doc_pos = self.p_encoder.partition_doc(d_pos_rep[i])
            indices_doc_neg = self.p_encoder.partition_doc(d_neg_rep[i])
            encoded_pos = combine(q_rep[i], d_pos_rep[i], indice_q_seg, indices_doc_pos)
            encoded_neg = combine(q_rep[i], d_neg_rep[i], indice_q_seg, indices_doc_neg)
            pos_entries.append(encoded_pos)
            neg_entries.append(encoded_neg)

        sample = {
            **rename_keys(pos_encoded, "pos"),
            **rename_keys(neg_encoded, "neg"),
            "teacher_p_score": s_pos, "teacher_n_score": s_neg}
        return {k: torch.tensor(v) for k, v in sample.items()}



#  tokenize into words
