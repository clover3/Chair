import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from list_lib import left, right
from ptorch.splade_tree.models.transformer_rep import Splade
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

class SpladeScorer:
    def __init__(self, model_type_or_dir):
        self.model = Splade(model_type_or_dir, agg="max")
        self.device = torch.device("cuda")
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}

    def get_rep(self, text, is_query):
        with torch.no_grad():
            input_ids = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = {k: v.to(self.device) for k, v in input_ids.items() if k not in {"id"}}

            if is_query:
                raw_rep = self.model(q_kwargs=input_ids, is_q=is_query)["q_rep"]
            else:
                raw_rep = self.model(d_kwargs=input_ids)["d_rep"]
            rep = raw_rep.squeeze()
            return rep

    def get_bow_rep(self, q_rep):
        col = torch.nonzero(q_rep).squeeze().cpu().tolist()

        weights = q_rep[col].cpu().tolist()
        d = {k: v for k, v in zip(col, weights)}
        sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        bow_rep = []
        for k, v in sorted_d.items():
            bow_rep.append((self.reverse_voc[k], round(v, 2)))
        return bow_rep

    def score_one(self, query: str, document: str) -> float:
        q_rep = self.get_rep(query, True)
        doc_rep = self.get_rep(document, False)
        score = torch.sum(q_rep * doc_rep, dim=-1)  #
        return score.tolist()

    def score(self, query: str, document: str) -> float:
        q_rep = self.get_rep(query, True)
        doc_rep = self.get_rep(document, False)
        score = torch.sum(q_rep * doc_rep, dim=-1)  #
        return score.tolist()

    def score_fn(self, qd_list: List[Tuple[str, str]]) -> List[float]:
        queries = left(qd_list)
        docs = right(qd_list)
        q_emb = self.get_rep(queries, True)
        d_emb = self.get_rep(docs, False)
        scores = torch.sum(q_emb * d_emb, dim=-1)
        scores = scores.cpu()
        scores = np.array(scores).tolist()
        return scores


def get_splade_as_reranker():
    model_type_or_dir = "distilsplade_max"
    model_type_or_dir = "/home/youngwookim_umass_edu/code/splade/distilsplade_max"
    scorer = SpladeScorer(model_type_or_dir)
    return scorer.score_fn


def main():
    score_fn = get_splade_as_reranker()
    queries = [
        "Where was Marie Curie born?",
        "Where was John Foley born?",
    ]
    sentences = [
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    ]
    qd = list(zip(queries, sentences))
    print(score_fn(qd))



    # demo()


if __name__ == "__main__":
    main()