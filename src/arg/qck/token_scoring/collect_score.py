import pickle
from collections import defaultdict
from typing import List, Iterable, Dict, NamedTuple, Tuple, Iterator

import numpy as np

from arg.qck.decl import qk_convert_map, QCKQuery, KDP
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.qck.token_scoring.decl import ScoreVector, TokenScore
from data_generator.tokenizer_wo_tf import get_word_level_location, get_tokenizer
from estimator_helper.output_reader import join_prediction_with_info
from exec_lib import run_func_with_config
from list_lib import lmap, dict_value_map
from misc_lib import group_by, TimeEstimator
from trainer.np_modules import sigmoid


class QKTokenLevelOutEntry(NamedTuple):
    logits: ScoreVector
    input_ids: List[int]
    query: QCKQuery
    kdp: KDP

    @classmethod
    def from_dict(cls, d):
        return QKTokenLevelOutEntry(d['logits'], d['input_ids'], d['query'], d['kdp'])


WordAsID = str


def ids_to_word_as_id(ids):
    return " ".join(map(str, ids))


def decode_word_as_id(tokenizer, word: WordAsID) -> List[str]:
    ids = map(int, word.split())
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return tokens


def collect_by_words(tokenizer, entry: QKTokenLevelOutEntry) -> Iterator[Tuple[WordAsID, TokenScore]]:
    input_ids = entry.input_ids
    probs = sigmoid(np.array(entry.logits))
    intervals: List[Tuple[int, int]] = get_word_level_location(tokenizer, input_ids)
    for st, ed in intervals:
        ids = input_ids[st:ed]
        word: WordAsID = ids_to_word_as_id(ids)
        scores = probs[st:ed]
        avg_scores = np.mean(scores, axis=0)
        yield word, avg_scores


def group_average_per_query(outputs) -> Dict[str, Dict[WordAsID, np.array]]:
    tokenizer = get_tokenizer()

    def collect_by_word_fn(entry: QKTokenLevelOutEntry):
        return collect_by_words(tokenizer, entry)

    print("Grouping entries")
    grouped: Dict[str, List[QKTokenLevelOutEntry]] = group_by(outputs, lambda x: x.query.query_id)

    def average_scores(out_entries: List[QKTokenLevelOutEntry]) -> Dict[WordAsID, np.array]:
        items: List[Iterable[Tuple[WordAsID, TokenScore]]] = lmap(collect_by_word_fn, out_entries)
        d: Dict[WordAsID, List] = defaultdict(list)
        for item in items:
            item: Iterable[Tuple[WordAsID, TokenScore]] = item
            for word, probs in item:
                d[word].append(probs)

        def average_per_dim(probs_list) -> np.array:
            return np.mean(np.array(probs_list), axis=0)

        out_d: Dict[WordAsID, np.array] = dict_value_map(average_per_dim, d)
        return out_d

    print("Collecting token level scores")
    per_query_infos: Dict[str, Dict[WordAsID, np.array]] = {}
    ticker = TimeEstimator(len(grouped))
    for key, value in grouped.items():
        per_query_infos[key] = average_scores(value)
        ticker.tick()

    return per_query_infos


def collect_and_save_score(config):
    info_path = config['info_path']
    pred_path = config['pred_path']
    save_path = config['save_path']

    info = load_combine_info_jsons(info_path, qk_convert_map, False)
    predictions: List[Dict] = join_prediction_with_info(pred_path,
                                                        info,
                                                        ['data_id', 'logits', 'input_ids', 'label_ids'],
                                                        )
    outputs: Iterable[QKTokenLevelOutEntry] = map(QKTokenLevelOutEntry.from_dict, predictions)

    per_query_infos: Dict[str, Dict[WordAsID, np.array]] = group_average_per_query(outputs)
    pickle.dump(per_query_infos, open(save_path, "wb"))


if __name__ == "__main__":
    run_func_with_config(collect_and_save_score)
