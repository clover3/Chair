from collections import OrderedDict
from typing import List, Dict, Any, Tuple

from data_generator.create_feature import create_int_feature, create_float_feature

from trainer_v2.per_project.transparency.splade_regression.data_loaders.regression_loader import XEncoded


def get_vector_regression_encode_fn_batched(max_seq_length):
    def pad_truncate(items, target_len) -> List[List]:
        truncated = [t[:target_len] for t in items]
        pad_len_list = [target_len - len(t) for t in truncated]
        padded_list = [item + [0] * pad_len for item, pad_len in zip(truncated, pad_len_list)]
        return padded_list

    def encode_batched(batch: Dict[int, List]) -> OrderedDict:
        X = batch[0]
        Y = batch[1]
        input_ids, attention_mask = zip(*X)
        indices_list, values_list = zip(*Y)
        max_len = max(map(len, input_ids))
        target_len = min(max_len, max_seq_length)
        input_ids_batched = pad_truncate(input_ids, target_len)
        attention_mask_batched = pad_truncate(attention_mask, target_len)

        # flatten the list of list
        def get_ragged_list_features(ll: List[List[Any]], prefix: str):
            flat_values = []
            len_info = []
            for l in ll:
                flat_values.extend(l)
                len_info.append(len(l))

            return {
                prefix + "_flat_values": flat_values,
                prefix + "_len_info": len_info,
            }

        # ll: List of List (because batched
        ll_rep: Dict[str, List[List]] = {
            "input_ids": input_ids_batched,
            "attention_mask": attention_mask_batched,
            'y_values': values_list,
            'y_indices': indices_list,
        }

        features = OrderedDict()
        for key, ll in ll_rep.items():
            if key == 'y_indices':
                # ll will be List of List[List[int]]
                reduced_l = 0
                new_ll = []
                for l in ll:
                    assert len(l) == 1
                    reduced_l = l[0]
                    new_ll.append(reduced_l)
                ll = new_ll
            d = get_ragged_list_features(ll, key)
            for key, value in d.items():

                if type(value[0]) == int:
                    features[key] = create_int_feature(value)
                elif type(value[0]) == float:
                    features[key] = create_float_feature(value)
                else:
                    print(key, type(value[0]))
                    raise Exception()

        # Keys:
        # y_values_flat_values, y_values_len_info, y_indices_flat_values, y_indices_len_info
        return features

    return encode_batched


def pad_truncate(seq, max_seq_length):
    seq = seq[:max_seq_length]
    pad_len = max_seq_length - len(seq)
    seq = seq + [0] * pad_len
    return seq


def get_vector_regression_encode_fn(max_text_seq_length, max_vector_indices):
    def encode_fn(item: Tuple[XEncoded, Any]) -> OrderedDict:
        X, Y = item
        input_ids, attention_mask = X
        assert len(input_ids) == len(attention_mask)
        input_ids = pad_truncate(input_ids, max_text_seq_length)
        attention_mask = pad_truncate(attention_mask, max_text_seq_length)
        indices, values = Y
        assert len(indices) == 1
        indices = indices[0]

        indices = pad_truncate(indices, max_vector_indices)
        values = pad_truncate(values, max_vector_indices)


        features = OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["attention_mask"] = create_int_feature(attention_mask)
        features["y_indices"] = create_int_feature(indices)
        features["y_values"] = create_float_feature(values)
        # Keys:
        # y_values_flat_values, y_values_len_info, y_indices_flat_values, y_indices_len_info
        return features

    return encode_fn


def get_three_text_encode_fn(max_text_seq_length):
    def encode_fn(three_item: Tuple[Dict]) -> OrderedDict:
        features = OrderedDict()
        for idx, item in enumerate(three_item):
            assert len(item['input_ids']) == len(item['attention_mask'])
            input_ids = pad_truncate(item['input_ids'], max_text_seq_length)
            attention_mask = pad_truncate(item['attention_mask'], max_text_seq_length)
            features[f"input_ids_{idx}"] = create_int_feature(input_ids)
            features[f"attention_mask_{idx}"] = create_int_feature(attention_mask)
        return features

    return encode_fn

