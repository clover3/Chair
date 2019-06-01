
import sys
from task.transformer_est import Transformer, Classification
from models.transformer import bert
from models.transformer import hyperparams
from data_generator.shared_setting import NLI
from trainer.tf_module import *
import tensorflow as tf
from data_generator.NLI import nli
from data_generator.ubuntu import ubuntu
from trainer.experiment import Experiment
from trainer.ExperimentConfig import ExperimentConfig




def train_nil():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLI_only_{}".format("B")
    e_config.num_epoch = 2
    e_config.save_interval = 30 * 60  # 30 minutes


    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    e.train_nli(nli_setting, e_config, data_loader)



def train_nil_on_bert():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "NLI_Only_{}".format("C")
    e_config.num_epoch = 2
    e_config.save_interval = 30 * 60  # 30 minutes


    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = None
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    #load_id = ("NLI_bert_w_explain", 'model-91531')
    #load_id = ("NLI_Only_A", "model-0")
    e.train_nli_ex_0(nli_setting, e_config, data_loader, load_id, False)



def train_snli_on_bert():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "SNLI_Only_{}".format("1")
    e_config.num_epoch = 1
    e_config.save_interval = 3 * 60 * 60  # 30 minutes


    data_loader = nli.SNLIDataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = None
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_nli_ex_0(nli_setting, e_config, data_loader, load_id, f_train_ex=False)

def test_snli():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "SNLIEx_Test"
    e_config.load_names = ['bert', 'cls_dense']  # , 'aux_conflict']
    data_loader = nli.SNLIDataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    todo = []
    load_id = ("SNLI_Only_A", 'model-0')
    todo.append(load_id)
    todo.append(("SNLI_Only_1", 'model-0'))


    for load_id in todo:
        tf.reset_default_graph()
        e.test_acc(nli_setting, e_config, data_loader, load_id)


def train_snli_ex():
    hp = hyperparams.HPBert()
    hp.compare_deletion_num = 20
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "SNLIEx_B"
    e_config.ex_val = False
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense'] #, 'aux_conflict']

    #explain_tag = 'match'  # 'dontcare'  'match' 'mismatch'

    #explain_tag = 'mismatch'
    #explain_tag = 'conflict'
    data_loader = nli.SNLIDataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLI_run_nli_warm", "model-97332")
    #load_id = ("NLIEx_A", "model-16910")
    #load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    #load_id = ("NLIEx_D", "model-1964")
    #load_id = ("NLIEx_D", "model-1317")
    load_id = ("SNLI_Only_A", 'model-0')
    e.train_nli_any_way(nli_setting, e_config, data_loader, load_id)


def train_mnli_any_way():
    hp = hyperparams.HPBert()
    hp.compare_deletion_num = 20
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "NLIEx_AnyA"
    e_config.ex_val = False
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense']  # , 'aux_conflict']

    # explain_tag = 'match'  # 'dontcare'  'match' 'mismatch'

    # explain_tag = 'mismatch'
    # explain_tag = 'conflict'
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_run_A", 'model-0')
    e.train_nli_any_way(nli_setting, e_config, data_loader, load_id)


def pred_snli_ex():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "SNLIEx_B"
    e_config.load_names = ['bert', 'cls_dense', 'aux_conflict']

    data_loader = nli.SNLIDataLoader(hp.seq_max, nli_setting.vocab_filename, True)

    load_id = ("SNLIEx_B", 'model-10275')
    e.predict_rf(nli_setting, e_config, data_loader, load_id, "test")





def train_nli_with_reinforce():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_{}".format("S")
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense'] #, 'aux_conflict']

    explain_tag = 'conflict'  # 'dontcare'  'match' 'mismatch'

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLI_run_nli_warm", "model-97332")
    #load_id = ("NLIEx_A", "model-16910")
    #load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    #load_id = ("NLIEx_D", "model-1964")
    #load_id = ("NLIEx_D", "model-1317")
    load_id = ("NLI_run_A", 'model-0')
    e.train_nli_ex_1(nli_setting, e_config, data_loader, load_id, explain_tag)


def train_pairing():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_{}".format("T")
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense']# 'aux_conflict']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLIEx_P_match", "model-1636")
    load_id = ("NLI_run_A", 'model-0')
    PAIRING_NLI = 6
    e.train_nli_smart(nli_setting, e_config, data_loader, load_id, 'match', PAIRING_NLI)


def train_nli_smart_rf():



    hp = hyperparams.HPSENLI()
    hp.compare_deletion_num = 20
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()

    e_config.name = "NLIEx_{}".format("match")
    e_config.num_epoch = 1
    e_config.ex_val = True
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense'] #, 'aux_conflict']

    #explain_tag = 'match'  # 'dontcare'  'match' 'mismatch'
    explain_tag = 'match'
    #explain_tag = 'mismatch'
    #explain_tag = 'conflict'

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLI_run_nli_warm", "model-97332")
    #load_id = ("NLIEx_A", "model-16910")
    #load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    #load_id = ("NLIEx_D", "model-1964")
    #load_id = ("NLIEx_D", "model-1317")
    load_id = ("NLI_run_A", 'model-0')

    e.train_nli_smart(nli_setting, e_config, data_loader, load_id, explain_tag, 5)





def tuning_train_nli_rf():
    for g_del in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tf.reset_default_graph()
        hp = hyperparams.HPSENLI()
        hp.g_val = g_del
        hp.compare_deletion_num = 20
        e = Experiment(hp)
        nli_setting = NLI()
        nli_setting.vocab_size = 30522
        nli_setting.vocab_filename = "bert_voca.txt"

        e_config = ExperimentConfig()

        e_config.name = "NLIEx_{}".format("match_del_{}".format(g_del))
        e_config.num_epoch = 1
        e_config.ex_val = True
        e_config.save_interval = 30 * 60  # 30 minutes
        e_config.load_names = ['bert', 'cls_dense'] #, 'aux_conflict']

        #explain_tag = 'match'  # 'dontcare'  'match' 'mismatch'
        explain_tag = 'conflict'
        #explain_tag = 'mismatch'
        #explain_tag = 'conflict'

        data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
        #load_id = ("NLI_run_nli_warm", "model-97332")
        #load_id = ("NLIEx_A", "model-16910")
        #load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
        #load_id = ("NLIEx_D", "model-1964")
        #load_id = ("NLIEx_D", "model-1317")
        load_id = ("NLI_run_A", 'model-0')

        e.train_nli_smart(nli_setting, e_config, data_loader, load_id, explain_tag, 5)




def train_nli_with_premade():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_{}".format("HB")
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert'] #, 'cls_dense'] #, 'aux_conflict']

    explain_tag = 'conflict'  # 'dontcare'  'match' 'mismatch'

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLI_run_nli_warm", "model-97332")
    #load_id = ("NLIEx_A", "model-16910")
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_nli_ex_with_premade_data(nli_setting, e_config, data_loader, load_id, explain_tag)


def do_test_dev_acc():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_Test"
    e_config.load_names = ['bert', 'cls_dense']#, 'aux_conflict']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLI_bare_A", 'model-195608')
    load_id = ("NLIEx_S", 'model-4417')
    load_id = ("NLIEx_Y_conflict", "model-9636")
    load_id = ("NLI_Only_C", 'model-0')

    e.test_acc(nli_setting, e_config, data_loader, load_id)


def analyze_nli_ex():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    explain_tag = 'match'

    e_config = ExperimentConfig()
    #e_config.name = "NLIEx_{}_premade_analyze".format(explain_tag)
    e_config.name = "NLIEx_{}_analyze".format(explain_tag)
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert' ,'cls_dense', 'aux_conflict']


    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLIEx_E_align", "model-23621")
    #load_id = ("NLIEx_I_match", "model-1238")

    if explain_tag == 'conflict':
        load_id = ("NLIEx_Y_conflict", "model-12039")
        #load_id = ("NLIEx_HB", "model-2684")
    elif explain_tag == 'match':
        load_id = ("NLIEx_P_match", "model-1636")
        load_id = ("NLIEx_X_match", "model-12238")
    elif explain_tag =='mismatch':
        load_id = ("NLIEx_U_mismatch", "model-10265")
    e.nli_visualization(nli_setting, e_config, data_loader, load_id, explain_tag)


def interactive():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIInterative"
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense'] #, 'aux_conflict']


    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_Only_B", 'model-0')

    e.nli_interactive_list(nli_setting, e_config, data_loader, load_id)

def interactive_visual():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIInterative"
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense', 'aux_conflict']
    load_id = ("NLIEx_U_mismatch", "model-10265")
    load_id = ("NLIEx_Y_conflict", 'model-12039')

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)

    e.nli_interactive_visual(nli_setting, e_config, data_loader, load_id)


def analyze_nli_pair():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_pair_analyze"
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert' ,'cls_dense', 'aux_conflict']


    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLIEx_T", "model-12097")
    e.nli_visualization_pairing(nli_setting, e_config, data_loader, load_id)



def baseline_explain():
    hp = hyperparams.HPBert()
    hp.batch_size *= 32
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("nli_warm")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense']

    explain_tag = 'conflict'  # 'dontcare'  'match' 'mismatch'
    #explain_tag = 'match'
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_run_A", 'model-0')
    e.nli_explain_baselines(nli_setting, e_config, data_loader, load_id, explain_tag)



def predict_rf():
    hp = hyperparams.HPBert()
    hp.batch_size = 256
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    target_label = 'match'
    #data_id = 'test_mismatch'
    data_id = "{}_1000".format(target_label)
    e_config = ExperimentConfig()

    del_g = 0.7
    e_config.name = "X_match_del_{}".format(del_g)
    e_config.load_names = ['bert', 'cls_dense', 'aux_conflict']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    #load_id = ("NLI_bare_A", 'model-195608')
    #load_id = ("NLIEx_O", 'model-10278')
    load_id = ("NLIEx_W_mismatch", "model-12030")
    load_id = ("NLIEx_Y_conflict", "model-12039")
    load_id = ("NLIEx_X_match", "model-12238")
    load_id = ("NLIEx_match_del_{}".format(del_g), "model-4390")
    e.predict_rf(nli_setting, e_config, data_loader, load_id, data_id)



def pred_mnli_anyway():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    e_config = ExperimentConfig()
    e_config.name = "NLIEx_AnyA"
    e_config.load_names = ['bert', 'cls_dense', 'aux_conflict']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    target_label = 'mismatch'
    data_id = "{}_1000".format(target_label)
    load_id = ("NLIEx_AnyA", 'model-2785')
    e.predict_rf(nli_setting, e_config, data_loader, load_id, data_id)

def attribution_predict():
    hp = hyperparams.HPBert()

    target_label = 'mismatch'
    e = Experiment(hp)
    hp.batch_size = 512
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("nli_eval")
    e_config.load_names = ['bert', 'cls_dense']

    #data_id = "test_{}".format(target_label)
    data_id = "{}_1000".format(target_label)
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_Only_C", 'model-0')
    e.nli_attribution_predict(nli_setting, e_config, data_loader, load_id, target_label, data_id)


def baseline_predict():
    hp = hyperparams.HPBert()
    hp.batch_size = 512
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("nli_eval")
    e_config.load_names = ['bert', 'cls_dense']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_Only_C", 'model-0')
    target_label =  "mismatch"
    data_id = "{}_1000".format(target_label)
    start = int(sys.argv[1])
    print(start)
    sub_range = (start, start + 100)
    #sub_range = None
    e.nli_baseline_predict(nli_setting, e_config, data_loader,
                          load_id, target_label, data_id, sub_range)



def predict_lime_snli():
    hp = hyperparams.HPBert()
    hp.batch_size = 1024 + 512 + 256
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "SNLI_LIME_{}".format("eval")
    e_config.load_names = ['bert', 'cls_dense']

    start = int(sys.argv[1])
    print("Begin", start)
    sub_range = (start, start + 100)

    data_loader = nli.SNLIDataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("SNLI_Only_A", 'model-0')
    e.predict_lime_snli(nli_setting, e_config, data_loader, load_id, "test", sub_range)



def predict_lime_snli_continue():
    hp = hyperparams.HPBert()
    hp.batch_size = 512 + 256
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "SNLI_LIME_{}".format("eval")
    e_config.load_names = ['bert', 'cls_dense']

    data_loader = nli.SNLIDataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("SNLI_Only_A", 'model-0')
    e.predict_lime_snli_continue(nli_setting, e_config, data_loader, load_id, "test")



def attribution_explain():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("nli_eval")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls_dense']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLI_run_nli_warm", "model-97332")
    load_id = ("NLI_Only_A", 'model-0')
    e.nli_attribution_baselines(nli_setting, e_config, data_loader, load_id)


def test_fidelity():
    hp = hyperparams.HPBert()
    e = Experiment(hp)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    is_senn = False

    e_config = ExperimentConfig()
    e_config.name = "NLIEx_{}".format("Fidelity")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    if is_senn:
        e_config.load_names = ['bert', 'cls_dense', 'aux_conflict']
    else:
        e_config.load_names = ['bert', 'cls_dense']
    explain_tag = 'conflict'

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    load_id = ("NLIEx_Y_conflict", 'model-12039')
    #load_id = ("NLI_Only_C", 'model-0')
    #e.eval_fidelity(nli_setting, e_config, data_loader, load_id, explain_tag)
    e.eval_fidelity_gradient(nli_setting, e_config, data_loader, load_id, explain_tag)




def train_nli_with_reinforce_old():
    hp = hyperparams.HPNLI2()
    e = Experiment(hp)
    nli_setting = NLI()

    e_config = ExperimentConfig()
    e_config.name = "NLI_run_{}".format("retest")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'dense_cls'] #, 'aux_conflict']

    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename)
    load_id = ("interval", "model-48040")
    e.train_nli_ex_0(nli_setting, e_config, data_loader, load_id, True)


def train_ubuntu():
    hp = hyperparams.HPUbuntu()
    hp.batch_size = 16
    e = Experiment(hp)
    voca_setting = NLI()
    voca_setting.vocab_size = 30522
    voca_setting.vocab_filename = "bert_voca.txt"

    e_config = ExperimentConfig()
    e_config.name = "Ubuntu_{}".format("A")
    e_config.num_epoch = 1
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'reg_dense'] #, 'aux_conflict']

    data_loader = ubuntu.DataLoader(hp.seq_max, voca_setting.vocab_filename, voca_setting.vocab_size, True)
    #load_id = ("NLI_run_nli_warm", "model-97332")
    #load_id = ("NLIEx_A", "model-16910")
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("Ubuntu_A", "model-5145")

    e.train_ubuntu(e_config, data_loader, load_id)


def ubuntu_train_gen():
    hp = hyperparams.HPUbuntu()
    e = Experiment(hp)
    voca_setting = NLI()
    voca_setting.vocab_size = 30522
    voca_setting.vocab_filename = "bert_voca.txt"

    data_loader = ubuntu.DataLoader(hp.seq_max, voca_setting.vocab_filename, voca_setting.vocab_size, True)
    for i in range(3,30):
        ubuntu.batch_encode(i)
    #e.gen_ubuntu_data(data_loader)


def test_ubuntu():
    hp = hyperparams.HPUbuntu()
    hp.batch_size = 16
    e = Experiment(hp)
    voca_setting = NLI()
    voca_setting.vocab_size = 30522
    voca_setting.vocab_filename = "bert_voca.txt"

    data_loader = ubuntu.DataLoader(hp.seq_max, voca_setting.vocab_filename, voca_setting.vocab_size, True)

    e.test_valid_ubuntu(data_loader)


if __name__ == '__main__':
    action = "predict_rf"
    locals()[action]()
