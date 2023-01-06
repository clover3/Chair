from dataset_specific.ufet.dev_parse import load_ufet, load_types
from trainer_v2.keras_server.name_short_cuts import get_pep_cache_client, NLIPredictorSig, get_pep_client
from trainer_v2.per_project.tli.qa_scorer.nli_direct import get_entail


def main():
    items = load_ufet("dev")
    type_list = load_types()
    predict_fn: NLIPredictorSig = get_pep_client()
    for e in items:
        payload = []
        s = " ".join(e.get_full_token())
        for t in type_list:
            payload.append((s, t))
        probs = predict_fn(payload)
        scores = list(map(get_entail, probs))

        pred = set()
        for s, t in zip(scores, type_list):
            if s > 0.7:
                pred.add(t)

        n_common = len(pred.intersection(e.y_str))
        prec = n_common / len(pred)
        recall = n_common / len(e.y_str)
        print("{0:.4f} {1:.4f}".format(prec, recall))


if __name__ == "__main__":
    main()