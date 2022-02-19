import functools
from typing import List, Callable, Dict

import nltk
import numpy as np

from cache import load_from_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import group_by
from tlm.qtype.analysis_fde.fde_module import FDEModuleEx
from tlm.qtype.analysis_fde.runner.build_q_emb_from_samples import load_q_bias
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import PartialSegment, SegJoinPolicyIF
from tlm.qtype.partial_relevance.complement_search_pckg.complement_search import FuncContentSegJoinPolicy
from tlm.qtype.partial_relevance.complement_search_pckg.query_vector import to_id_format
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance
from tlm.qtype.partial_relevance.mmd_cached_client import get_mmd_cache_client
from tlm.qtype.partial_relevance.segmented_text import SegmentedText
from tlm.qtype.wikify.doc_span_count import word_count_per_ft, print_log_odd_per_span
from trainer.np_modules import sigmoid


def get_fde_module_ex(run_name):
    q_embedding_d: Dict[str, np.array] = load_from_pickle("{}_q_embedding".format(run_name))
    q_bias_d: Dict[str, np.array] = load_q_bias(run_name)
    cluster: List[int] = load_from_pickle("{}_clusters".format(run_name))
    return FDEModuleEx(q_embedding_d, q_bias_d, cluster)


def get_payload_list(seg_join_policy: SegJoinPolicyIF,
                     si: SegmentedInstance,
                     seg_index_to_keep,
                     candidates: List[PartialSegment]) \
        -> List[SegmentedInstance]:
    payload_list: List[SegmentedInstance] = []
    for c in candidates:
        new_si = get_new_segmented_instance(seg_join_policy, seg_index_to_keep, si, c)
        payload_list.append(new_si)
    return payload_list


def get_new_segmented_instance(seg_join_policy, seg_index_to_keep, si, c):
    new_text1 = seg_join_policy.join_tokens(si.text1, c, seg_index_to_keep)
    new_si: SegmentedInstance = SegmentedInstance(new_text1, si.text2)
    return new_si


def get_amherst_wiki_doc():
    doc = "Amherst is a town in Hampshire County, Massachusetts, United States, in the Connecticut River valley. " \
          "As of the 2020 census, the population was 39,263,[5] " \
          "making it the highest populated municipality in Hampshire County (although the county seat is Northampton). " \
          "The town is home to Amherst College, Hampshire College, and the University of Massachusetts Amherst, " \
          "three of the Five Colleges. " \
          "The name of the town is pronounced without the h (\"AM-erst\") by natives and long-time residents,[6] " \
          "giving rise to the local saying, \"only the 'h' is silent\", " \
          "in reference both to the pronunciation and to the town's politically active populace. " \
          "Amherst has three census-designated places: Amherst Center, North Amherst, and South Amherst. " \
          "Amherst is part of the Springfield, Massachusetts Metropolitan Statistical Area. " \
          "Lying 22 miles (35 km) north of the city of Springfield, " \
          "Amherst is considered the northernmost town in the Hartford-Springfield Metropolitan Region, " \
          "\"The Knowledge Corridor\". Amherst is also located in the Pioneer Valley, " \
          "which encompasses Hampshire, Hampden and Franklin " \
          "counties."
    return doc


def get_insulin_doc():
    doc = "where does insulin come from ? \" health diseases & conditions diabetes where does insulin come from ? some diabetics have to inject themselves with insulin . where do the pharmaceutical companies get it from ? is it made ? ? ? answers appreciated thank you follow 11 answers answers relevance rating newest oldest best answer : insulin is a hormone produced by beta cells in the pancreas . it has three important functions : 1 allow glucose to pass into cells , where it is used for energy . suppress excess production of sugar in the liver and muscles . suppress breakdown of fat for energy . in the absence of insulin , blood sugar levels rise because muscle and fat cells aren ' t able to utilize glucose for energy . they signal the body that they ' re \" \" hungry . \" \" the liver then releases glycogen , a form of stored glucose . this further increases the blood sugar level . when the blood sugar level reaches about 180 mg / dl , glucose begins to spill into the urine . large amounts of water are needed to dissolve the excess sugar , resulting in excessive thirst and urination . without glucose for energy , the body begins to metabolize protein and fat . fat metabolism results in the production of ketones in the liver . ketones are excreted in the urine along with sodium bicarbonate , which results in a decrease in the p h of the blood . this condition is called acidosis . to correct the acidosis , the body begins a deep , labored respiration , called kussmaul ' s respiration . left unchecked , a person in this situation will fall into a coma and die . common questions why do i have to inject insulin ? insulin must be injected because it is a protein . if it were taken orally , the body ' s digestive system would break it down , rendering it useless . where should i store insulin ? unopened insulin vials should be kept cool . storing them in the refrigerator will help them last as long as possible . never freeze insulin , however , as freezing can destroy it . open insulin , whether vials or pens , can be kept at room temperature for about a month . where does insulin come from ? insulin used by people with diabetes can come from three sources : human ( created via recombinant dna methods ) , pork , or beef . beef insulin has been discontinued in the us , and essentially all people who are newly diagnosed are placed on human insulin . what kinds of insulin are there"
    return doc


def get_sentences():
    doc = get_amherst_wiki_doc()
    return nltk.sent_tokenize(doc)


def get_entity_doc():
    doc = get_amherst_wiki_doc()
    entity = "Amherst"
    return entity, doc


def get_n_relevant(entity, doc, fde_module, forward_fn):
    tokenizer = get_tokenizer()

    def enc(text) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    entity_ids = enc(entity)
    doc_ids = enc(doc)
    indices_entity_ids = list(range(0, len(entity_ids)))
    seg1 = SegmentedText(entity_ids, [[], indices_entity_ids])
    seg2 = SegmentedText.from_tokens_ids(doc_ids)
    seg_instance = SegmentedInstance(seg1, seg2)
    scores: np.array = fde_module.compute_score_from_ids(entity_ids, doc_ids)
    threshold = 0.5
    probs = sigmoid(scores)
    fde_based_relevant = np.less(threshold, probs)
    candidate_indices = [idx for idx, r in enumerate(fde_based_relevant) if r]
    n_candidates = len(candidate_indices)
    # Convert candidates to
    candidate_func_spans: List[str] = [fde_module.func_span_list[idx] for idx in candidate_indices]
    tokenizer = get_tokenizer()
    to_id_format_fn = functools.partial(to_id_format, tokenizer)
    candidates: List[PartialSegment] = list(map(to_id_format_fn, candidate_func_spans))
    seg_join_policy = FuncContentSegJoinPolicy()
    preserve_seg_idx = 1
    payload_list: List[SegmentedInstance] = get_payload_list(seg_join_policy, seg_instance, preserve_seg_idx,
                                                             candidates)
    direct_scores: List[float] = forward_fn(payload_list)
    span_to_score: Dict[str, float] = {s: direct_scores[idx] for idx, s in enumerate(candidate_func_spans)}
    valid_complement_list_s: List[str] = [s for s in candidate_func_spans if span_to_score[s] >= threshold]
    n_relevant = len(valid_complement_list_s)
    return n_candidates, n_relevant


def display_per_cluster(entity: str, doc: str, fde_module, forward_fn):
    tokenizer = get_tokenizer()
    def enc(text: str) -> List[int]:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    entity_ids = enc(entity)
    doc_ids = enc(doc)
    indices_entity_ids = list(range(0, len(entity_ids)))
    seg1 = SegmentedText(entity_ids, [[], indices_entity_ids])
    seg2 = SegmentedText.from_tokens_ids(doc_ids)
    seg_instance = SegmentedInstance(seg1, seg2)
    scores: np.array = fde_module.compute_score_from_ids(entity_ids, doc_ids)
    threshold = 0.5
    probs = sigmoid(scores)
    fde_based_relevant = np.less(threshold, probs)
    candidate_indices = [idx for idx, r in enumerate(fde_based_relevant) if r]
    # Convert candidates to
    candidate_func_spans: List[str] = [fde_module.func_span_list[idx] for idx in candidate_indices]
    tokenizer = get_tokenizer()
    to_id_format_fn = functools.partial(to_id_format, tokenizer)
    candidates: List[PartialSegment] = list(map(to_id_format_fn, candidate_func_spans))
    seg_join_policy = FuncContentSegJoinPolicy()
    preserve_seg_idx = 1
    payload_list: List[SegmentedInstance] = get_payload_list(seg_join_policy, seg_instance, preserve_seg_idx,
                                                             candidates)
    direct_scores: List[float] = forward_fn(payload_list)
    span_to_score: Dict[str, float] = {s: direct_scores[idx] for idx, s in enumerate(candidate_func_spans)}
    valid_complement_list_s: List[str] = [s for s in candidate_func_spans if span_to_score[s] >= threshold]
    n_relevant = len(valid_complement_list_s)
    cluster_grouped = group_by(valid_complement_list_s, fde_module.get_cluster_id)
    for cluster_id, items in cluster_grouped.items():
        cluster_size = len(fde_module.cluster_id_to_idx[cluster_id])
        # print("Cluster {}".format(cluster_id))
        n_rel = len(items)
        print(f"{cluster_id}\t{n_rel}\t{cluster_size}")
        # items.sort(key=lambda s: span_to_score[s], reverse=True)
        # for s in items:
        #     print(f"{s}: {span_to_score[s]:.2f}")


def per_sentences():
    doc = get_amherst_wiki_doc()
    entity = "Amherst"
    entity = "insulin"
    doc = get_insulin_doc()
    run_name = "qtype_2Y_v_train_120000"
    fde_module: FDEModuleEx = get_fde_module_ex(run_name)
    sents = nltk.sent_tokenize(doc)
    cache_client = get_mmd_cache_client("localhost")
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = cache_client.predict
    for s in [doc] + sents:
        n_candidates, n_relevant = get_n_relevant(entity, s, fde_module, forward_fn)
        print("{}\t{}".format(n_candidates, n_relevant))


def do_display_per_cluster():
    doc = get_amherst_wiki_doc()
    sents = get_sentences()
    entity = "Amherst"
    run_name = "qtype_2Y_v_train_120000"
    fde_module: FDEModuleEx = get_fde_module_ex(run_name)
    cache_client = get_mmd_cache_client("localhost")
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = cache_client.predict
    s = sents[0]
    display_per_cluster(entity, s, fde_module, forward_fn)


def do_perturbation():
    doc = get_amherst_wiki_doc()
    entity = "Amherst"
    tokenizer = get_tokenizer()

    run_name = "qtype_2Y_v_train_120000"
    fde_module: FDEModuleEx = get_fde_module_ex(run_name)
    per_span_dict, tf = word_count_per_ft(fde_module, entity, doc)
    print_log_odd_per_span(per_span_dict, tf)

    # TODO: build segment in word level
    #       for each word,
    #           delete and get vector
    #           count change of relevant func_span
    #           Count [(word, func_span)]
    # List of Changed spans
    fde_span_doc_summary: Dict[int, List[str]] = NotImplemented
    fde_span_sent_summary = NotImplemented


if __name__ == "__main__":
    do_perturbation()
