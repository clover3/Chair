import argparse
import json
import sys
import warnings

from cache import load_from_pickle

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#from arg.perspectives.mz_model import KNRMEx, AvgEmbedding
import matchzoo as mz
import tensorflow as tf
import arg.perspectives.mz_dataset

from misc_lib import tprint


def load_data_pack():
    train_pack = arg.perspectives.mz_dataset.load_data('train', task='classification')
    valid_pack = arg.perspectives.mz_dataset.load_data('dev', task='classification')
    return train_pack, valid_pack


def load_data_pack_wiki():
    train_pack = mz.datasets.wiki_qa.load_data('train', task='classification')
    valid_pack = mz.datasets.wiki_qa.load_data('dev', task='classification')
    return train_pack, valid_pack


def knrm_processed():
    train_pack, valid_pack = load_data_pack()
    preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=20, fixed_length_right=20,
                                                      remove_stop_words=False)

    train_pack_processed = preprocessor.fit_transform(train_pack)
    valid_pack_processed = preprocessor.transform(valid_pack)
    return preprocessor, train_pack_processed, valid_pack_processed


def drmm_processed():
    train_pack = mz.datasets.wiki_qa.load_data('train', task='classification')
    valid_pack = mz.datasets.wiki_qa.load_data('dev', task='classification')
    test_pack = mz.datasets.wiki_qa.load_data('test', task='classification')

    preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=100,
                                                      remove_stop_words=False)
    train_pack_processed = preprocessor.fit_transform(train_pack)
    dev_pack_processed = preprocessor.transform(valid_pack)
    test_pack_processed = preprocessor.transform(test_pack)
    return preprocessor, train_pack_processed, dev_pack_processed


def parse_arg():
    parser = argparse.ArgumentParser(description='File should be stored in ')
    parser.add_argument("--config_path", default="")
    parsed = parser.parse_args(sys.argv[1:])
    return parsed


def get_drmm_model(preprocessor, task, output_dim):
    bin_size = 30
    model = mz.models.DRMM()
    model.params.update(preprocessor.context)
    model.params['input_shapes'] = [[10, ], [10, bin_size, ]]
    model.params['task'] = task
    model.params['mask_value'] = 0
    model.params['embedding_output_dim'] = output_dim
    model.params['mlp_num_layers'] = 1
    model.params['mlp_num_units'] = 10
    model.params['mlp_num_fan_out'] = 1
    model.params['mlp_activation_func'] = 'tanh'
    model.params['optimizer'] = 'adadelta'
    model.build()
    model.compile()
    #model.backend.summary()
    return model


def prepare_embedding(output_dim, term_index):
    # glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=output_dim)
    # embedding_matrix = glove_embedding.build_matrix(term_index)
    # save_to_pickle(embedding_matrix, "embedding_matrix")
    # # normalize the word embedding for fast histogram generating.
    # l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    # embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
    embedding_matrix = load_from_pickle("embedding_matrix")
    return embedding_matrix



def prepare_model_and_data(param):
    tprint("Loading data")
    preprocessor, train_processed, valid_processed = drmm_processed()
    print(train_processed)
    tprint("Defining task")
    classification_task = mz.tasks.classification.Classification()
    classification_task.metrics = ['accuracy']
    output_dim = 300
    tprint('output_dim : {}'.format(output_dim))
    # Initialize the model, fine-tune the hyper-parameters.
    tprint("building model")
    #model = mz.models.KNRM()
    #model = KNRMEx()
    # model = AvgEmbedding()
    # model.params.update(preprocessor.context)
    # model.params['task'] = classification_task
    # model.params['embedding_output_dim'] = output_dim
    # model.params['embedding_trainable'] = False
    # model.params['kernel_num'] = 11
    # model.params['sigma'] = 0.1
    # model.params['exact_sigma'] = 0.001
    glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=output_dim)

    model = get_drmm_model(preprocessor, classification_task, output_dim)
    for key, v in param.items():
        if key in model.params:
            model.params[key] = v

    model.guess_and_fill_missing_params(verbose=1)

    step_per_epoch = 423 * 128
    num_max_steps = 100 * step_per_epoch

    if 'lr_decay' in param and param['lr_decay']:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=param['lr'],
            decay_steps=num_max_steps / 20,
            decay_rate=0.9)
    else:
        lr = param['lr']
    model.params['optimizer'] = tf.keras.optimizers.Adam(learning_rate=lr)

    model.build()
    model.compile()
    tprint("processing embedding")
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = prepare_embedding(output_dim, term_index)
    model.load_embedding_matrix(embedding_matrix)
    return model, train_processed, valid_processed



def main2():
    args = parse_arg()
    param = json.load(open(args.config_path, "r"))
    tprint("Loading data")
    preprocessor, train_processed, valid_processed = drmm_processed()
    print(train_processed)
    tprint("Defining task")
    classification_task = mz.tasks.classification.Classification()
    classification_task.metrics = ['accuracy']
    output_dim = 300
    tprint('output_dim : {}'.format(output_dim))
    # Initialize the model, fine-tune the hyper-parameters.
    tprint("building model")
    model = get_drmm_model(preprocessor, classification_task, output_dim)
    # for key, v in param.items():
    #     if key in model.params:
    #         model.params[key] = v
    #
    #model.guess_and_fill_missing_params(verbose=1)

    step_per_epoch = 423 * 128
    num_max_steps = 100 * step_per_epoch
    #
    # if 'lr_decay' in param and param['lr_decay']:
    #     lr = tf.keras.optimizers.schedules.ExponentialDecay(
    #         initial_learning_rate=param['lr'],
    #         decay_steps=num_max_steps / 20,
    #         decay_rate=0.9)
    # else:
    #     lr = param['lr']
    # model.params['optimizer'] = tf.keras.optimizers.Adam(learning_rate=lr)

    model.build()
    model.compile()
    tprint("processing embedding")
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = prepare_embedding(output_dim, term_index)
    model.load_embedding_matrix(embedding_matrix)

    hist_callback = mz.data_generator.callbacks.Histogram(embedding_matrix, bin_size=30, hist_mode='LCH')
    tprint("defining generator")
    train_generator = mz.DataGenerator(train_processed, batch_size=param['batch_size'],
                                       shuffle=True, callbacks=[hist_callback])

    valid_x, valid_y = valid_processed.unpack()
    evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=len(valid_x))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=1, mode='min')
    tprint("fitting")
    callbacks = [evaluate]

    # if param['early_stop']:
    #     callbacks.append(early_stop)
    history = model.fit_generator(train_generator, epochs=100, callbacks=callbacks, workers=30,
                                  use_multiprocessing=True)

def main():
    args = parse_arg()
    param = json.load(open(args.config_path, "r"))
    model, train_processed, valid_processed = prepare_model_and_data(param)

    hist_callback = mz.data_generator.callbacks.Histogram(embedding_matrix, bin_size=30, hist_mode='LCH')
    tprint("defining generator")
    train_generator = mz.DataGenerator(train_processed, batch_size=param['batch_size'],
                                       shuffle=True, callbacks=[hist_callback])

    valid_x, valid_y = valid_processed.unpack()
    evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=len(valid_x))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=1, mode='min')
    tprint("fitting")
    callbacks = [evaluate]

    if param['early_stop']:
        callbacks.append(early_stop)
    history = model.fit_generator(train_generator, epochs=100, callbacks=callbacks, workers=5,
                                  use_multiprocessing=False)


if __name__ == "__main__":
    main2()
