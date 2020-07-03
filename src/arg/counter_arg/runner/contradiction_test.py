from typing import List, Callable

import numpy as np

from arg.counter_arg.contradiction_requester import Request
from arg.counter_arg.eval import collect_failure, EvalCondition
from arg.counter_arg.header import Passage
from arg.counter_arg.methods import bm25_predictor
from arg.counter_arg.methods.bm25_predictor import get_bm25_module
from arg.counter_arg.methods.stance_query import get_stance_check_candidate
from arg.perspectives.collection_based_classifier import NamedNumber


def work():
    query = "A presidential position enable the democratic selection of a head-of-state The alternative to the monarch is obvious. Many states around the world have Presidential systems, either like the United States where the President fulfils both the role of the Head of State and the Head of Government combining the two roles. Or as in Italy or Germany where there is both a head of state (usually president) and a head of government (usually Prime Minister, although Germany’s is Chancellor) where the head of state is respected but is mostly a ceremonial role. Finally there may be both a head of state and head of government where both are powerful as in France. Therefore the head of state can still be in whatever role the state requires. Most importantly in all these cases the head of state is elected rather than simply gaining the position on account of birth."
    gold = "The head of government will already be elected. There is no need to create a competing centre of power that has the same popular legitimacy. Just as there are worries that an elected house of lords would want more powers due to its new found legitimacy an elected head of state could demand the same. Such a change would be disruptive and is not necessary."
    pred = "That’s equally an argument against international criminal law as head of state immunity. While there may be instances where the head of state or government has to take decisions that might be criminal for the greater good – for example ordering the abduction or assassination of a terrorist – these instances are rare and most of the time the courts will take into account the good as well as the bad. However there are equally times where it is good that someone fears prosecution, if they do it is a sign that what they are doing is wrong. Bombing of Germany could have ended when all military targets had been hit, it need not have involved incendiary bombing of civilian targets. In Japan there was a third option of accepting a conditional surrender – one that guaranteed the position of the Emperor, since the Allies ultimately agreed this anyway there would have been little loss."
    split = "training"

    scorer: Callable[[Passage, List[Passage]], List[NamedNumber]] = bm25_predictor.get_scorer(split)

    cont_requester = Request()
    bm25 = get_bm25_module("training")
    def pairwise(query, candidate):
        sent_term_list_q = get_stance_check_candidate(query, bm25)
        sent_term_list_c = get_stance_check_candidate(candidate, bm25)

        r = []
        for sent_q, terms_q in sent_term_list_q:
            for sent_c, terms_c in sent_term_list_c:
                r.append((sent_q, sent_c))
        return r

    def print_result_list(result_list):
        for e in result_list:
            sent1, sent2, scores = e
            pred = np.argmax(scores)
            if pred == 2:
                print("Sent1: ", sent1)
                print("Sent2: ", sent2)
                print(scores)

    def count_cont(result_list):
        num_cont = sum([1 for _, _, scores in result_list if np.argmax(scores) == 2])
        return num_cont

    c_g_list = []
    c_p_list = []
    for idx, e in enumerate(collect_failure(split, scorer, EvalCondition.EntirePortalCounters)):
        if idx == 30:
            break
        try:
            print(idx)
            query, gold, pred = e
            result_list = cont_requester.request_multiple(pairwise(query, gold))
            cnt = count_cont(result_list)
            print("Gold", cnt)
            print_result_list(result_list)
            c_g_list.append(cnt)
            result_list = cont_requester.request_multiple(pairwise(query, pred))
            cnt = count_cont(result_list)
            print("Pred", cnt)
            print_result_list(result_list)
            c_p_list.append(cnt)
        #print_result_list(result_list)
        except Exception as e :
            print(e)

    print(c_g_list)
    print(c_p_list)


if __name__ == "__main__":
    work()
