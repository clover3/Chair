import math
from collections import Counter, defaultdict
from typing import Tuple, Dict, List

from data_generator.tokenizer_wo_tf import ids_to_text
from misc_lib import get_second
from tlm.qtype.analysis_fde.fde_module import FDEModuleEx
from tlm.qtype.partial_relevance.runner.sent_tokenize_dev import sentence_segment_w_indices
from tlm.qtype.partial_relevance.segmented_text import SegmentedText


def word_count_per_ft(fde_module: FDEModuleEx, entity: str, doc: str) -> Tuple[Dict[str, Counter], Counter]:
    tokenizer = fde_module.tokenizer

    def enc(text) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    max_seq_length = 512
    entity_ids = enc(entity)
    max_doc_len = max_seq_length - len(entity_ids) - 3
    doc_ids = enc(doc)
    doc_ids = doc_ids[:max_doc_len]
    ids, indices = sentence_segment_w_indices(tokenizer, doc_ids)
    text2 = SegmentedText(ids, indices)
    base_promising_span = fde_module.get_promising_from_ids(entity_ids, text2.tokens_ids)
    per_span_dict = defaultdict(Counter)
    tf = Counter()
    for seg_idx in text2.enum_seg_idx():
        text2_new = text2.get_dropped_text([seg_idx])
        promising_spans = fde_module.get_promising_from_ids(entity_ids, text2_new.tokens_ids)
        affected_func_spans = [s for s in base_promising_span if s not in promising_spans]
        seg_text = ids_to_text(tokenizer, text2.get_tokens_for_seg(seg_idx))
        for word in seg_text.split():
            tf[word] += 1
            for span in affected_func_spans:
                per_span_dict[span][word] += 1
    return per_span_dict, tf


def print_log_odd_per_span(per_span_dict, tf):
    ctf = sum(tf.values())
    min_tf = 80
    for func_span, counter_per_fs in per_span_dict.items():
        # P(w|func_span)
        n_sum = sum(counter_per_fs.values())
        log_odd_list = []
        candidate_words = [w for w in counter_per_fs if tf[w] >= min_tf]
        for w in candidate_words:
            cnt = counter_per_fs[w]
            p_w_func_span = cnt / n_sum
            log_odd = math.log(p_w_func_span) - math.log(tf[w] / ctf)
            log_odd_list.append((w, log_odd))
        log_odd_list.sort(key=get_second, reverse=True)
        print("-----")
        print(func_span)
        for w, log_odd in log_odd_list[:10]:
            cnt = counter_per_fs[w]
            print("{}\t{}\t{}".format(w, cnt, log_odd))
