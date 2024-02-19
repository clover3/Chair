import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from list_lib import left, right
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

# Compute token embeddings
def get_contriever_like_encoder(model_name='facebook/contriever'):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    # Mean pooling
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            inputs.to(device)
            outputs = model(**inputs)
            return mean_pooling(outputs[0], inputs['attention_mask'])
    return encode


def get_scorer_from_dual_encoder(encoder):
    def score_fn(qd_list: List[Tuple[str, str]]) -> List[float]:
        queries = left(qd_list)
        docs = right(qd_list)
        q_emb = encoder(queries)
        d_emb = encoder(docs)
        scores = torch.sum(q_emb * d_emb, dim=-1)
        scores = scores.cpu()
        scores = np.array(scores).tolist()
        return scores

    return score_fn


def get_contriever_as_reranker(model_name='facebook/contriever'):
    encoder = get_contriever_like_encoder(model_name)
    score_fn = get_scorer_from_dual_encoder(encoder)
    return score_fn


def main():
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')

    queries = [
        "Where was Marie Curie born?",
        "Where was John Foley born?",
    ]
    sentences = [
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    ]

    qd = zip(queries, sentences)
    score_fn = get_contriever_as_reranker()

    print(score_fn(qd))

    # # Apply tokenizer
    #
    # encode = get_contriever_like_encoder()
    # q_emb = encode(queries)
    # d_emb = encode(sentences)
    #
    # n_sent = len(sentences)
    # scores = torch.sum(q_emb.unsqueeze(1) * d_emb.unsqueeze(0), dim=-1)
    # scores = scores.cpu()
    #
    # for i in range(len(queries)):
    #     print(queries[i])
    #     print(scores[i])


if __name__ == "__main__":
    main()
