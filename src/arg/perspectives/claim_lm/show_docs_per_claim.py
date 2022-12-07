import random
from collections import Counter
from typing import List, Dict, Set

from arg.perspectives.claim_lm.passage_common import iterate_passages
from arg.perspectives.clueweb_db import load_doc
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids, get_claim_perspective_id_dict
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms
from base_type import FilePath
from cache import save_to_pickle
from datastore.interface import preload_man
from datastore.table_names import TokenizedCluewebDoc
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from list_lib import lmap, flatten, left, lfilter, right
from misc_lib import average, two_digit_float
from models.classic.lm_util import get_lm_log, subtract, smooth, average_counters
from models.classic.stopword import load_stopwords_for_query
from visualize.html_visual import HtmlVisualizer, Cell


def show_docs_per_claim():
    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claims = claims[:10]
    top_n = 10
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)

    preload_docs(ranked_list, claims, top_n)

    show_docs(claims, ranked_list, top_n)


def show_missing():
    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claims = claims[:10]
    top_n = 100
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)

    preload_docs(ranked_list, claims, top_n)
    report_missing(claims, ranked_list, top_n)


def show_docs(claims, ranked_list, top_n):
    # for each claim
    for c in claims:
        q_res: List[SimpleRankedListEntry] = ranked_list[str(c['cId'])]
        print(c['cId'], c['text'])
        for i in range(top_n):
            try:
                doc = load_doc(q_res[i].doc_id)
                print()
                print("Doc rank {}".format(i))
                print(" ".join(doc))
            except KeyError:
                pass
        print("--------")


def report_missing(claims, ranked_list, top_n):
    # for each claim
    n_missing = 0
    for c in claims:
        q_res: List[SimpleRankedListEntry] = ranked_list[str(c['cId'])]
        print(c['cId'], c['text'])
        missing = []
        for i in range(top_n):
            try:
                doc = load_doc(q_res[i].doc_id)
            except KeyError:
                missing.append(i)
                pass
        print(missing)
        n_missing += len(missing)

    print("")


stopwords = load_stopwords_for_query()
def get_cell_from_token(token, log_odd):
    if token in stopwords:
        log_odd = 0

    if log_odd > 0:
        s = min(150, log_odd * 50)
        c = Cell(token, s, target_color="B")
    elif log_odd < 0:
        s = min(150, -log_odd * 50)
        c = Cell(token, s, target_color="R")
    else:
        c = Cell(token)
    return c

def preload_docs(ranked_list, claims, top_n):
    def get_doc_ids(claim: Dict):
        # Find the q_res
        q_res: List[SimpleRankedListEntry] = ranked_list[str(claim['cId'])]
        return list([q_res[i].doc_id for i in range(top_n)])

    all_doc_ids: Set[str] = set(flatten(lmap(get_doc_ids, claims)))
    print(f"total of {len(all_doc_ids)} docs")
    print("Accessing DB")
    #  Get the doc from DB
    preload_man.preload(TokenizedCluewebDoc, all_doc_ids)


def join_docs_and_lm():
    gold = get_claim_perspective_id_dict()

    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claims = claims[:10]
    top_n = 10
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    preload_docs(ranked_list, claims, top_n)
    claim_lms = build_gold_lms(claims)
    claim_lms_d = {lm.cid: lm for lm in claim_lms}
    bg_lm = average_counters(lmap(lambda x: x.LM, claim_lms))
    log_bg_lm = get_lm_log(bg_lm)


    stopwords.update([".",",", "!", "?"])

    alpha = 0.1

    html_visualizer = HtmlVisualizer("doc_lm_joined.html")

    def get_cell_from_token2(token, probs):
        if token.lower() in stopwords:
            probs = 0
        probs = probs * 1e5
        s = min(100, probs)
        c = Cell(token, s)
        return c

    tokenizer = KrovetzNLTKTokenizer()
    for c in claims:
        q_res: List[SimpleRankedListEntry] = ranked_list[str(c['cId'])]
        html_visualizer.write_headline("{} : {}".format(c['cId'], c['text']))

        clusters: List[List[int]] = gold[c['cId']]

        for cluster in clusters:
            html_visualizer.write_paragraph("---")
            p_text_list: List[str] = lmap(perspective_getter, cluster)
            for text in p_text_list:
                html_visualizer.write_paragraph(text)
            html_visualizer.write_paragraph("---")
        claim_lm = claim_lms_d[c['cId']]
        topic_lm_prob = smooth(claim_lm.LM, bg_lm, alpha)
        log_topic_lm = get_lm_log(smooth(claim_lm.LM, bg_lm, alpha))
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)

        s = "\t".join(left(log_odd.most_common(30)))
        html_visualizer.write_paragraph("Log odd top: "+ s)
        not_found = set()

        def get_log_odd(x):
            x = tokenizer.stemmer.stem(x)
            if x not in log_odd:
                not_found.add(x)
            return log_odd[x]

        def get_probs(x):
            x = tokenizer.stemmer.stem(x)
            if x not in topic_lm_prob:
                not_found.add(x)
            return topic_lm_prob[x]

        for i in range(top_n):
            try:
                doc = load_doc(q_res[i].doc_id)
                cells = lmap(lambda x: get_cell_from_token(x, get_log_odd(x)), doc)
                html_visualizer.write_headline("Doc rank {}".format(i))
                html_visualizer.multirow_print(cells, width=20)
            except KeyError:
                pass
        html_visualizer.write_paragraph("Not found: {}".format(not_found))




def doc_lm_scoring():
    gold = get_claim_perspective_id_dict()

    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claims = claims
    top_n = 10
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    preload_docs(ranked_list, claims, top_n)
    claim_lms = build_gold_lms(claims)
    claim_lms_d = {lm.cid: lm for lm in claim_lms}
    bg_lm = average_counters(lmap(lambda x: x.LM, claim_lms))
    log_bg_lm = get_lm_log(bg_lm)

    stopwords = load_stopwords_for_query()
    alpha = 0.5

    html_visualizer = HtmlVisualizer("doc_lm_doc_level.html")

    tokenizer = KrovetzNLTKTokenizer()
    random_passages = []
    num_pos_sum = 0
    num_pos_exists = 0
    for c in claims:
        q_res: List[SimpleRankedListEntry] = ranked_list[str(c['cId'])]
        html_visualizer.write_headline("{} : {}".format(c['cId'], c['text']))
        # for cluster in clusters:
        #     html_visualizer.write_paragraph("---")
        #     p_text_list: List[str] = lmap(perspective_getter, cluster)
        #     for text in p_text_list:
        #         html_visualizer.write_paragraph(text)
        #     html_visualizer.write_paragraph("---")
        claim_lm = claim_lms_d[c['cId']]
        topic_lm_prob = smooth(claim_lm.LM, bg_lm, alpha)
        log_topic_lm = get_lm_log(smooth(claim_lm.LM, bg_lm, alpha))
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)

        claim_text = c['text']
        claim_tokens = tokenizer.tokenize_stem(claim_text)

        scores = []
        for t in claim_tokens:
            if t in log_odd:
                scores.append(log_odd[t])
        threshold = average(scores)

        s = "\t".join(left(log_odd.most_common(30)))
        html_visualizer.write_paragraph("Log odd top: "+ s)
        not_found = set()

        def get_log_odd(x):
            x = tokenizer.stemmer.stem(x)
            if x not in log_odd:
                not_found.add(x)
            return log_odd[x]

        def get_probs(x):
            x = tokenizer.stemmer.stem(x)
            if x not in topic_lm_prob:
                not_found.add(x)
            return topic_lm_prob[x]

        def get_passage_score(p):
            return sum([log_odd[tokenizer.stemmer.stem(t)] for t in p]) / len(p) if len(p) > 0 else 0

        passages = iterate_passages(q_res, top_n, get_passage_score)

        passages.sort(key=lambda x: x[1], reverse=True)
        html_visualizer.write_paragraph("Threshold {}".format(threshold))

        top5_scores = right(passages[:5])
        bot5_scores = right(passages[-5:])

        if len(random_passages) > 5:
            random_sel_pssages = random.choices(random_passages, k=5)
        else:
            random_sel_pssages = []
        random5_scores = lmap(get_passage_score, random_sel_pssages)

        def score_line(scores):
            return " ".join(lmap(two_digit_float, scores))
        html_visualizer.write_paragraph("top 5: " + score_line(top5_scores))
        html_visualizer.write_paragraph("bot 5: " + score_line(bot5_scores))
        html_visualizer.write_paragraph("random 5: " + score_line(random5_scores))

        num_pos = len(lfilter(lambda x: x[1] > 0, passages))
        num_pos_sum += num_pos
        if num_pos > 0:
            num_pos_exists += 1

        def print_doc(doc, html_visualizer, score):
            cells = lmap(lambda x: get_cell_from_token(x, get_log_odd(x)), doc)
            html_visualizer.write_headline("score={}".format(score))
            html_visualizer.multirow_print(cells, width=20)

        random_passages.extend(left(passages))
        if threshold < 0:
            continue
        for doc, score in passages:
            if score < 0:
                break
            print_doc(doc, html_visualizer, score)

        html_visualizer.write_headline("Bottom 5")
        for doc, score in passages[-5:]:
            print_doc(doc, html_visualizer, score)

    print("{} claims. {} docs on {} claims".format(len(claims), num_pos_sum, num_pos_exists))


def a_relevant():
    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claims = claims
    top_n = 10
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    preload_docs(ranked_list, claims, top_n)
    claim_lms = build_gold_lms(claims)
    claim_lms_d = {lm.cid: lm for lm in claim_lms}
    bg_lm = average_counters(lmap(lambda x: x.LM, claim_lms))
    log_bg_lm = get_lm_log(bg_lm)

    stopwords = load_stopwords_for_query()
    alpha = 0.3

    tokenizer = KrovetzNLTKTokenizer()
    all_passages = []
    entries = []
    for c in claims:
        q_res: List[SimpleRankedListEntry] = ranked_list[str(c['cId'])]
        claim_lm = claim_lms_d[c['cId']]
        log_topic_lm = get_lm_log(smooth(claim_lm.LM, bg_lm, alpha))
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)

        claim_text = c['text']
        claim_tokens = tokenizer.tokenize_stem(claim_text)

        scores = []
        for t in claim_tokens:
            if t in log_odd:
                scores.append(log_odd[t])
        base = average(scores)

        def get_passage_score(p):
            def get_score(t):
                if t in stopwords:
                    return 0
                return log_odd[tokenizer.stemmer.stem(t)]

            return sum([get_score(t) for t in p]) / len(p) if len(p) > 0 else 0

        passages = iterate_passages(q_res, top_n, get_passage_score)

        all_passages.extend(passages)
        a_rel_passages = lfilter(lambda x: x[1] > 0, passages)

        entries.append((c, a_rel_passages))

    data = entries, all_passages

    save_to_pickle(data, "pc_train_a_passages")


def main():
    #show_missing()
    doc_lm_scoring()
    # ls


if __name__ == "__main__":
    main()