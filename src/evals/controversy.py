
from models.controversy import *
from models.bert_controversy import BertPredictor
from evaluation import compute_auc, compute_pr_auc, compute_acc, AP

def eval_all_contrv():
    ams_X, ams_Y = amsterdam.get_dev_data(False)
    clue_X, clue_Y = controversy.load_clueweb_testset()
    guardian_X, guardian_Y = controversy.load_guardian()



    models = []
    models.append(("Amsterdam", get_wiki_doc_lm()))
    models.append(("BertAms", BertPredictor()))
    models.append(("MH16", get_dbpedia_contrv_lm()))
    models.append(("Guardian", get_guardian16_lm()))
    # models.append(("Guardian2", get_guardian_selective_lm()))

    test_sets = [("Ams18", [ams_X, ams_Y]),
                 ("Clueweb" ,[clue_X, clue_Y]),
                 ("Guardian", [guardian_X, guardian_Y])]

    for set_name, test_set in test_sets:
        dev_X, dev_Y = test_set
        print(set_name)
        for name, model in models:
            scores = model.score(dev_X)
            auc = compute_pr_auc(scores, dev_Y)
            print("{}\t{}".format(name, auc))

