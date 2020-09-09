from collections import defaultdict
from typing import Dict, List, Tuple, Iterable

import scipy.special

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.ppnc.ppnc_decl import PayloadAsTokens
from base_type import FilePath
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lfilter, flatten
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def read_passage_scores(prediction_file,
                        info: Dict,
                        recover_subtokens
                        ) \
        -> Dict[int, List[Dict]] :
    data = EstimatorPredictionViewer(prediction_file)
    print("Num data ", data.data_len)
    output: Dict[int, List] = defaultdict(list)
    for entry in data:
        logits = entry.get_vector("logits")
        data_id = entry.get_vector("data_id")[0]
        try:
            cur_info = info[int(data_id)]
            cid = cur_info['cid']
            d = {
                'cid': cid,
                'passage': recover_subtokens(entry.get_vector("input_ids")),
                'logits': logits,
                'data_id': data_id,
            }
            output[cid].append(d)
        except KeyError as e:
            print("Key error")
            print("data_id", data_id)
            pass
    return output

def get_recover_subtokens():
    tokenizer = get_tokenizer()

    def recover_subtokens(input_ids) -> List[str]:
        tokens1, tokens2 = split_p_h_with_input_ids(input_ids, input_ids)
        return tokenizer.convert_ids_to_tokens(tokens2)

    return recover_subtokens


def join_perspective(cid_outputs: List[Tuple[int, List[Dict]]],
                     candidate_perspectives: Dict[int, List[Dict]])\
        -> Iterable[Tuple[int, int, List[Dict]]]:

    for cid, passages in cid_outputs:
        for perspective in candidate_perspectives[cid]:
            pid = perspective['pid']
            yield cid, pid, passages


def collect_good_passages(data_id_to_info: Dict[int, Dict],
                          passage_score_path: FilePath,
                          config: Dict
                          ):
    recover_subtokens = get_recover_subtokens()

    score_cut = config['score_cut']
    top_k = config['top_k']
    grouped_scores: Dict[int, List[Dict]] = read_passage_scores(passage_score_path, data_id_to_info, recover_subtokens)

    def get_score_from_logit(logits):
        return scipy.special.softmax(logits)[1]

    def is_good(d: Dict):
        score = get_score_from_logit(d['logits'])
        return score >= score_cut

    output = []
    num_passges = []
    for cid, passages in grouped_scores.items():
        good_passages = lfilter(is_good, passages)
        good_passages.sort(key=lambda d: get_score_from_logit(d['logits']), reverse=True)
        num_passges.append(len(good_passages))
        if good_passages:
            output.append((cid, good_passages[:top_k]))
        else:
            scores = list([get_score_from_logit(d['logits']) for d in passages])
            scores.sort(reverse=True)

    print(num_passges)
    print("{} of {} claims has passages".format(len(output), len(grouped_scores)))
    return output


def put_texts(joined_payload: Iterable[Tuple[int, int, List[Dict]]],
              claims: List[Dict],
              tokenizer,
              data_id_man
              ) \
        -> Iterable[PayloadAsTokens]:
    cid_to_text = {c['cId']: c['text'] for c in claims}

    cache_tokenize = {}
    def tokenize(text):
        if text in cache_tokenize:
            return cache_tokenize[text]
        tokens = tokenizer.tokenize(text)
        cache_tokenize[text] = tokens
        return tokens

    def encode(e: Tuple[int, int, List[Dict]]):
        cid, pid, passages = e
        text1 = tokenize(cid_to_text[cid])
        text2 = tokenize(perspective_getter(pid))

        for passage_idx, passage in enumerate(passages):
            info = {
                'cid': cid,
                'pid': pid,
                'passage_idx': passage_idx,
                'passage': passage['passage'],
                'c_text': cid_to_text[cid],
                'p_text': perspective_getter(pid)
            }
            yield PayloadAsTokens(passage=passage['passage'],
                                   text1=text1,
                                   text2=text2,
                                   data_id=data_id_man.assign(info),
                                   is_correct=0
                                   )

    return flatten(map(encode, joined_payload))


