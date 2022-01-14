from typing import List

import numpy as np

from explain.pairing.predict import FlattenBatch


def predict_middle_in(sess, batches, logits,
                      loss_tensor, per_layer_logits_tensor, other_tensor_list,
                      embedding_feed_dict):
    num_layer = len(per_layer_logits_tensor)
    all_per_layer_logits: List[List[np.array]] = []
    flatten_batch = FlattenBatch()
    for batch in batches:
        input_ids, input_mask, segment_ids, hidden_vector, attention_mask = batch
        fet_tensors = [logits, loss_tensor, per_layer_logits_tensor] + other_tensor_list
        out_values \
            = sess.run(fet_tensors,
                       feed_dict=embedding_feed_dict(batch)
                       )
        logits_val, loss_val, per_layer_logits = out_values[:3]
        other_values = out_values[3:]
        all_per_layer_logits.append(per_layer_logits)
        d = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'logits': logits_val,
            'other_values': other_values
        }
        for j, value in enumerate(other_values):
            d["other_value_{}".format(j)] = value

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
        "logits": concat_d['logits'],
        "per_layer_logits": logits_grouped_by_layer
    }
    for j, _ in enumerate(other_tensor_list):
        key = "other_value_{}".format(j)
        output_d[key] = concat_d[key]
    return output_d


def predict_middle_out(sess, dev_batches, logits,
               loss_tensor, per_layer_logits_tensor, other_tensor_list,
               batch2feed_dict):
    num_layer = len(per_layer_logits_tensor)
    all_per_layer_logits: List[List[np.array]] = []
    flatten_batch = FlattenBatch()
    for batch in dev_batches:
        input_ids, input_mask, segment_ids, label = batch

        fet_tensors = [logits, loss_tensor, per_layer_logits_tensor] + other_tensor_list
        out_values \
            = sess.run(fet_tensors,
                       feed_dict=batch2feed_dict(batch)
                       )
        logits_val, loss_val, per_layer_logits = out_values[:3]
        other_values = out_values[3:]
        all_per_layer_logits.append(per_layer_logits)
        d = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'label': label,
            'logits': logits_val,
        }
        for j, value in enumerate(other_values):
            print(value.shape)
            d["other_value_{}".format(j)] = value

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
        "logits": concat_d['logits'],
        "label": concat_d['label'],
        "per_layer_logits": logits_grouped_by_layer
    }
    for j, _ in enumerate(other_tensor_list):
        key = "other_value_{}".format(j)
        output_d[key] = concat_d[key]
    return output_d