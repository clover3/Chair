from collections import Counter
from typing import Dict

from adhoc.bm25_class import BM25
from arg.perspectives import es_helper
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import claims_to_dict, get_claim_perspective_id_dict
from arg.perspectives.random_walk.pc_predict import normalize_counter
from list_lib import dict_value_map, lmap


def pc_predict_to_inspect(bm25_module: BM25,
                          q_tf_replace: Dict[int, Counter],
                          q_tf_replace_0: Dict[int, Counter],
                          claims,
                          top_k):
    gold = get_claim_perspective_id_dict()
    q_tf_replace_norm = dict_value_map(normalize_counter, q_tf_replace)
    q_tf_replace_0_norm = dict_value_map(normalize_counter, q_tf_replace_0)

    cid_to_text: Dict[int, str] = claims_to_dict(claims)
    c_qtf_d = {}
    for cid, c_text in cid_to_text.items():
        c_tokens = bm25_module.tokenizer.tokenize_stem(c_text)
        c_qtf_d[cid] = Counter(c_tokens)

    def counter_to_str(c: Dict) -> str:
        s = ""
        for k, v in c.items():
            s += "{0} {1:.2f}".format(k, v) + "\t"
        return s

    for claim in claims:
        cid = claim['cId']
        i_claim_id = int(cid)
        claim_text = claim['text']
        lucene_results = es_helper.get_perspective_from_pool(claim_text, 50)
        candidate_pids = []
        for rank, (_text, _pid, _score) in enumerate(lucene_results):
            candidate_pids.append(_pid)

        if i_claim_id in q_tf_replace_norm:
            claim_qtf = Counter(dict_value_map(lambda x: x * 1, c_qtf_d[i_claim_id]))
            ex_qtf = q_tf_replace_norm[i_claim_id]
            ex_qtf = Counter(dict(ex_qtf.most_common(50)))
            qtf = ex_qtf + claim_qtf
        else:
            qtf = c_qtf_d[i_claim_id]

        ranked_list = []
        for pid in candidate_pids:
            p_text = perspective_getter(int(pid))
            p_tokens = bm25_module.tokenizer.tokenize_stem(p_text)
            score = bm25_module.score_inner(qtf, Counter(p_tokens))
            debug_str = ""

            e = score, pid, p_text, debug_str
            ranked_list.append(e)

        gold_pids = gold[cid]

        def is_correct(pid):
            for pids in gold_pids:
                if pid in pids:
                    return True
            return False

        ranked_list.sort(key=lambda x:x[0], reverse=True)

        qtf_idf_applied = {k: v * bm25_module.term_idf_factor(k) for k, v in qtf.items()}
        print()
        print("Claim: ", cid, claim_text)
        for cluster in gold_pids:
            print("-")
            for pid in cluster:
                print(pid, perspective_getter(pid))
        print()
        print("qtf:", counter_to_str(qtf))
        if i_claim_id in q_tf_replace_norm and i_claim_id in q_tf_replace_0_norm:
            print("ex_qtf:", counter_to_str(ex_qtf))
            ex_qtf_0 = q_tf_replace_0_norm[i_claim_id]
            ex_qtf_0 = Counter(dict(ex_qtf_0.most_common(50)))
            print("ex_qtf_0:", counter_to_str(ex_qtf_0))
        print("qtf idf apllied:", counter_to_str(qtf_idf_applied))

        for score, pid, p_text, debug_str in ranked_list[:top_k]:

            if i_claim_id in q_tf_replace_0_norm:
                p_text = perspective_getter(int(pid))
                p_tokens = bm25_module.tokenizer.tokenize_stem(p_text)
                ex_qtf_0 = q_tf_replace_0_norm[i_claim_id]
                qtf = ex_qtf_0 + c_qtf_d[i_claim_id]
                score2 = bm25_module.score_inner(qtf, Counter(p_tokens))
                correct_str = "Y" if is_correct(pid) else "N"
                print("{0} {1:.2f} ({2:.2f}) {3} / {4} / {5}".format(correct_str, score, score2, p_text, score.name, score2.name))


def predict_see_candidate(bm25_module: BM25,
                          claims,
                          top_k):
    cid_to_text: Dict[int, str] = claims_to_dict(claims)
    c_qtf_d = {}
    for cid, c_text in cid_to_text.items():
        c_tokens = bm25_module.tokenizer.tokenize_stem(c_text)
        c_qtf_d[cid] = Counter(c_tokens)


    output = []
    for claim in claims:
        cid = claim['cId']
        claim_text = claim['text']
        lucene_results = es_helper.get_perspective_from_pool(claim_text, 50)
        candidate_pids = []
        for rank, (_text, _pid, _score) in enumerate(lucene_results):
            candidate_pids.append(_pid)

        p_text = lmap(perspective_getter, candidate_pids)

        p_tokens = lmap(bm25_module.tokenizer.tokenize_stem, p_text)

        acc_counter = Counter()
        for tokens in p_tokens[:30]:
            for t in tokens:
                acc_counter[t] += 1/len(tokens)
        c = normalize_counter(acc_counter)
        c_tokens = bm25_module.tokenizer.tokenize_stem(claim_text)
        qtf = Counter(c_tokens)
        qtf = c + qtf

        ranked_list = []
        for pid in candidate_pids:
            p_tokens = bm25_module.tokenizer.tokenize_stem(perspective_getter(pid))
            score = bm25_module.score_inner(qtf, Counter(p_tokens))
            ranked_list.append((pid, score))

        ranked_list.sort(key=lambda x: x[1], reverse=True)
        prediction_list = []

        for pid, score in ranked_list[:top_k]:
            p_entry = {
                'cid': cid,
                'pid': pid,
                'claim_text': claim_text,
                'perspective_text': perspective_getter(pid),
                'rationale': score.name,
                'score': score,
            }
            prediction_list.append(p_entry)
        output.append((cid, prediction_list))

    return output
