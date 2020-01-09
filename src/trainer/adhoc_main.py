import sys

import cpath
from data_generator.adhoc import score_loader
from data_generator.adhoc import ws
from data_generator.adhoc.weaksupervision import data_sampler
from data_generator.adhoc.weaksupervision.data_sampler import *
from models.transformer import hyperparams
from trainer.ExperimentConfig import ExperimentConfig
from trainer.experiment import Experiment


def train_adhoc_with_reinforce():
    hp = hyperparams.HPAdhoc()
    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}".format("E")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.train_adhoc(e_config, data_loader, load_id)



def train_adhoc_on_robust():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 16 * 3
    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}".format("FAD")
    e_config.num_epoch = 4
    e_config.save_interval = 10 * 60  # 60 minutes
    e_config.load_names = ['bert'] #, 'reg_dense']
    vocab_size = 30522

    data_loader = data_sampler.DataLoaderFromFile(hp.batch_size, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    #load_id = ("Adhoc_I2", 'model-290')
    e.train_adhoc2(e_config, data_loader, load_id)


def train_adhoc512():
    hp = hyperparams.HPFAD()
    hp.batch_size = 16
    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_J_{}".format("512")
    e_config.num_epoch = 4
    e_config.save_interval = 10 * 60  # 60 minutes
    e_config.load_names = ['bert', 'reg_dense']
    vocab_size = 30522

    data_loader = data_sampler.DataLoaderFromFile(hp.batch_size, vocab_size, 171)
    #load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("Adhoc_J_512", 'model-6189')
    e.train_adhoc2(e_config, data_loader, load_id)


def predict_adhoc512():
    hp = hyperparams.HPFAD()
    hp.batch_size = 16
    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_J_{}".format("512")
    e_config.num_epoch = 4
    e_config.save_interval = 10 * 60  # 60 minutes
    e_config.load_names = ['bert', 'reg_dense']
    vocab_size = 30522

    payload_path = os.path.join(cpath.data_path, "robust_payload", "enc_payload_512.pickle")
    task_idx = int(sys.argv[2])
    print(task_idx)
    load_id = ("Adhoc_J_512", 'model-6180')
    e.predict_robust(e_config, vocab_size, load_id, payload_path, task_idx)


def train_adhoc_fad():
    hp = hyperparams.HPFAD()
    hp.batch_size = 16
    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}".format("FAD")
    e_config.num_epoch = 4
    e_config.save_interval = 10 * 60  # 60 minutes
    e_config.load_names = ['bert'] #, 'reg_dense']
    vocab_size = 30522

    data_loader = data_sampler.DataLoaderFromFile(hp.batch_size, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    #load_id = ("Adhoc_I2", 'model-290')
    e.train_adhoc2(e_config, data_loader, load_id)




def train_adhoc_ex_on_robust():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 16
    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}".format("L")
    e_config.num_epoch = 4
    e_config.save_interval = 10 * 60  # 60 minutes
    e_config.load_names = ['bert', 'reg_dense']
    vocab_size = 30522

    data_loader = data_sampler.DataLoaderFromFile(hp.batch_size, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("Adhoc_J", 'model-9475')
    e.train_adhoc_ex(e_config, data_loader, load_id)



def predict_adhoc_robust_J():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}_eval".format("J")
    e_config.load_names = ['bert', 'reg_dense']
    vocab_size = 30522
    payload_path = os.path.join(cpath.data_path, "robust_payload", "payload_B_200.pickle")
    #payload_path = os.path.join(cpath.data_path, "robust_payload", "payload_desc.pickle")
    task_idx = int(sys.argv[2])
    print(task_idx)
    load_id = ("Adhoc_J", 'model-9475')
    e.predict_robust(e_config, vocab_size, load_id, payload_path, task_idx)



def predict_adhoc_robust_K():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}_eval".format("K")
    e_config.load_names = ['bert', 'dense1', 'dense_reg']
    vocab_size = 30522
    #payload_path = os.path.join(cpath.data_path, "robust_payload", "payload_B_200.pickle")
    payload_path = os.path.join(cpath.data_path, "robust_payload", "payload_desc.pickle")
    task_idx = int(sys.argv[2])
    print(task_idx)
    load_id = ("Adhoc_K", 'model-6397')
    e.predict_robust(e_config, vocab_size, load_id, payload_path, task_idx)



def predict_adhoc_robust_L():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}_eval".format("L")
    #e_config.load_names = ['bert', 'reg_dense']
    e_config.load_names = ['bert', 'reg_dense', 'aux_q_info']
    vocab_size = 30522
    payload_path = os.path.join(cpath.data_path, "robust_payload", "payload_B_200.pickle")
    task_idx = int(sys.argv[2])
    print(task_idx)

    q_id_list = [
        (301, 325),
        (326, 350),
        (351, 375),
        (376, 400),
        (401, 425),
        (426, 450),
        (601, 625),
        (626, 650),
        (651, 675),
        (676, 700),
    ]
    st, ed = q_id_list[task_idx]

    load_id = ("Adhoc_L", 'model-644')
    middle_result = e.predict_robust_L_part1(e_config, vocab_size, load_id, payload_path, (st, ed))

    preload_id2 = ("MergerE_C (copy)", 'model-3075')
    e.predict_robust_L_part2(e_config, middle_result, preload_id2, (st,ed) )


def predict_bm25_robust():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)
    e.rank_robust_bm25()


def run_adhoc_rank_on_robust():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}_eval".format("F")
    e_config.num_epoch = 4
    e_config.load_names = ['bert', 'reg_dense']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = data_sampler.DataLoaderFromFile(hp.batch_size, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("Adhoc_E", 'model-58338')
    e.rank_adhoc(e_config, data_loader, load_id)


def run_adhoc_rank():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}_eval2".format("E")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'reg_dense']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    load_id = ("Adhoc_E", 'model-58338')
    e.rank_adhoc(e_config, data_loader, load_id)


def run_ql_rank():
    hp = hyperparams.HPQL()

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}".format("D")
    e_config.num_epoch = 4
    e_config.save_interval = 30 * 60  # 30 minutes
    e_config.load_names = ['bert', 'cls']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.rank_ql(e_config, data_loader, load_id)


def test_ql():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}".format("C")
    e_config.num_epoch = 4
    e_config.load_names = ['bert', 'cls']
    vocab_size = 30522
    vocab_filename = "bert_voca.txt"

    data_loader = ws.DataLoader(hp.seq_max, vocab_filename, vocab_size)
    load_id = ("uncased_L-12_H-768_A-12", 'bert_model.ckpt')
    e.test_ql(e_config, data_loader, load_id)


def train_score_merger():
    hp = hyperparams.HPMerger_BM25()
    e = Experiment(hp)
    e_config = ExperimentConfig()
    e_config.name = "Merger_{}".format("A")
    e_config.num_epoch = 4

    data_loader = score_loader.DataLoader(hp.seq_max, hp.hidden_units)
    e.train_score_merger(e_config, data_loader)


def train_score_merger_on_vector():
    hp = hyperparams.HPMerger()
    e = Experiment(hp)
    e_config = ExperimentConfig()
    e_config.name = "MergerE_{}".format("E")

    data_loader = score_loader.NetOutputLoader(hp.seq_max, hp.hidden_units, hp.batch_size)
    e.train_score_merger(e_config, data_loader)


def pool_adhoc():
    hp = hyperparams.HPAdhoc()
    hp.batch_size = 512

    e = Experiment(hp)

    e_config = ExperimentConfig()
    e_config.name = "Adhoc_{}_pool".format("L")
    #e_config.load_names = ['bert', 'reg_dense']
    e_config.load_names = ['bert', 'reg_dense', 'aux_q_info']
    vocab_size = 30522
    task_idx = int(sys.argv[2])
    print(task_idx)
    payload_path = os.path.join(cpath.data_path, "robust", "robust_train_merge", "merger_train_{}.pickle".format(task_idx))

    load_id = ("Adhoc_L", 'model-644')
    e.predict_for_pooling(e_config, vocab_size, load_id, payload_path)



if __name__ == '__main__':
    action = sys.argv[1]
    locals()[action]()
