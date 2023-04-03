from misc_lib import get_first, get_second
from trainer_v2.per_project.transparency.splade_regression.modeling.splade_predictor import get_splade, SPLADEWrap, get_sparse_representation


def main():
    splade = get_splade()
    def get_text_rep(rep):
        r = get_sparse_representation(rep)
        return splade.decode_spare_rep(r)

    text_list = ["Androgen receptor define", "what is paranoid sc",
                 "  what is priority pass", "what is operating system misconfiguration"]
    # text_list = [ "how long is a day on ven", "how long is a typical car loan?",
    #               "turkey and china time difference",
    #               "how long is flight from hyd to dubai"]
    reps = splade.encode_batch(text_list)
    for text, rep in zip(text_list, reps):
        rep = get_text_rep(rep)
        rep.sort(key=get_second, reverse=True)
        simpler = " ".join(["{0}, {1:.1f}".format(token, score) for token, score in rep if score > 0.1])
        print(text, simpler)



if __name__ == "__main__":
    main()