
from cpath import data_path
from data_generator.tokenizer_b import FullTokenizerWarpper
from evaluation import *
from models.cnn_predictor import CNNPredictor
from models.controversy import *


def eval_all_contrv():
    ams_X, ams_Y = amsterdam.get_dev_data(False)
    clue_X, clue_Y = controversy.load_clueweb_testset()
    guardian_X, guardian_Y = controversy.load_guardian()



    models = []
    #models.append(("CNN/Wiki", CNNPredictor("WikiContrvCNN")))
    models.append(("CNN/Wiki", CNNPredictor("WikiContrvCNN_sigmoid", "WikiContrvCNN")))
    #models.append(("tlm/wiki", get_wiki_doc_lm()))
    #models.append(("Bert/Wiki", BertPredictor("WikiContrv2009")))
    #models.append(("Bert/Wiki", BertPredictor("WikiContrv2009_only_wiki")))
    #models.append(("tlm/dbpedia", get_dbpedia_contrv_lm()))
    #models.append(("tlm/Guardian", get_guardian16_lm()))
    #models.append(("yw_may", get_yw_may()))
    #models.append(("Guardian2", get_guardian_selective_lm()))

    test_sets = []
    #test_sets.append(("Ams18", [ams_X, ams_Y]))
    test_sets.append(("Clueweb" ,[clue_X, clue_Y]))
    #test_sets.append(("Guardian", [guardian_X, guardian_Y]))


    for set_name, test_set in test_sets:
        dev_X, dev_Y = test_set
        print(set_name)
        for name, model in models:
            scores = model.score(dev_X)
            auc = compute_pr_auc(scores, dev_Y)
            #auc = compute_auc(scores, dev_Y)
            acc = compute_opt_acc(scores, dev_Y)
            prec = compute_opt_prec(scores, dev_Y)
            recall = compute_opt_recall(scores, dev_Y)
            f1 = compute_opt_f1(scores, dev_Y)
            print("{0}\t{1:.03f}\t{2:.03f}\t{3:.03f}\t{4:.03f}\t{5:.03f}".format(name, auc, prec, recall, f1, acc))


def dataset_stat():
    ams_X, ams_Y = amsterdam.get_dev_data(False)
    clue_X, clue_Y = controversy.load_clueweb_testset()
    guardian_X, guardian_Y = controversy.load_guardian()

    vocab_size = 30522
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    encoder = FullTokenizerWarpper(voca_path)
    test_sets = []
    test_sets.append(("Ams18", [ams_X, ams_Y]))
    test_sets.append(("Clueweb" ,[clue_X, clue_Y]))
    test_sets.append(("Guardian", [guardian_X, guardian_Y]))

    for set_name, test_set in test_sets:
        dev_X, dev_Y = test_set
        num_over_size = 0
        length_list = []
        for doc in dev_X:
            tokens = encoder.encode(doc)
            if len(tokens) > 200:
                num_over_size += 1
            length_list.append(len(tokens))

        print("{0} {1:.03f} {2:.03f}".format(set_name, num_over_size / len(dev_X), average(length_list)))


if __name__ == '__main__':
    eval_all_contrv()
