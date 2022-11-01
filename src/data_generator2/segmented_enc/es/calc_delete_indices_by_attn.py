import pickle

from data_generator2.segmented_enc.es.path_helper import get_evidence_selected0_path
from dataset_specific.mnli.mnli_reader import MNLIReader
from misc_lib import TimeEstimator
from trainer_v2.custom_loop.attention_helper.attention_extractor import AttentionExtractor
from trainer_v2.custom_loop.attention_helper.evidence_selector_0 import get_delete_indices, \
    SegmentedPair2
from trainer_v2.custom_loop.attention_helper.model_shortcut import load_nli14_attention_extractor
from trainer_v2.custom_loop.attention_helper.runner.attn_reasonable import enum_segmentations
import numpy as np


def do_for_split(attn_extractor, split):
    reader = MNLIReader()
    nli_pair_iter = reader.load_split(split)
    ticker = TimeEstimator(reader.get_data_size(split), sample_size=100)
    batch_size = 16

    def iter_batches():
        batch = []
        for e in enum_segmentations(nli_pair_iter):
            batch.append(e)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    output = []
    for e_batch in iter_batches():
        payload = []
        for e in e_batch:
            payload.append((e.p_tokens, e.h_tokens))

        res = attn_extractor.predict_list(payload)

        for i, _ in enumerate(payload):
            e = e_batch[i]
            attn_score: np.array = res[i]
            delete_indices_list = get_delete_indices(attn_score, e)

            e_out = SegmentedPair2(e.p_tokens, e.h_tokens, e.st, e.ed,
                                   delete_indices_list[0], delete_indices_list[1],
                                   e.nli_pair)
            output.append(e_out)
            ticker.tick()
    save_path = get_evidence_selected0_path(split)
    pickle.dump(output, open(save_path, "wb"))


def main():
    attn_extractor: AttentionExtractor = load_nli14_attention_extractor()
    for split in ["dev", "train"]:
        do_for_split(attn_extractor, split)


if __name__ == "__main__":
    main()