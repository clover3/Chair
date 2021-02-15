from typing import Dict, List

import numpy as np


class FlattenBatch:
    def __init__(self):
        self.dict = {}
        self.is_first_batch = True

    def push(self, d: Dict):
        for key, value in d.items():
            if key not in self.dict:
                self.dict[key] = []
                if not self.is_first_batch:
                    print("WARNING this is not first batch but key {} is first observed".format(key))
            self.dict[key].append(value)

        self.is_first_batch = False

    def get_concatenated(self):
        out_d = {}
        for key in self.dict:
            out_d[key] = np.concatenate(self.dict[key], axis=0)
        return out_d


def predict_fn(sess, dev_batches,
                        loss_tensor, ex_scores_tensor, per_layer_logits_tensor,
                        batch2feed_dict):
    num_layer = len(per_layer_logits_tensor)
    all_per_layer_logits: List[List[np.array]] = []
    flatten_batch = FlattenBatch()
    for batch in dev_batches:
        input_ids, input_mask, segment_ids, label = batch
        loss_val, ex_scores, per_layer_logits \
            = sess.run([loss_tensor, ex_scores_tensor, per_layer_logits_tensor],
                       feed_dict=batch2feed_dict(batch)
                       )
        all_per_layer_logits.append(per_layer_logits)
        d = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'label': label,
            'ex_scores': ex_scores
        }

        for layer_no in range(num_layer):
            key = "per_layer_logits_{}".format(layer_no)
            d[key] = per_layer_logits[layer_no]
        flatten_batch.push(d)

    concat_d = flatten_batch.get_concatenated()
    logits_grouped_by_layer = []
    for layer_no in range(num_layer):
        t = np.concatenate([batch[layer_no] for batch in all_per_layer_logits], axis=0)
        logits_grouped_by_layer.append(t)

    output_d = {
        'input_ids': concat_d['input_ids'],
        "segment_ids": concat_d['segment_ids'],
        "input_mask": concat_d['input_mask'],
        "ex_scores": concat_d['ex_scores'],
        "label": concat_d['label'],
        "logits": logits_grouped_by_layer
    }
    return output_d