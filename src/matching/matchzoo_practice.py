import matchzoo as mz
import numpy as np

from cache import load_from_pickle


def get_processed_data():
    train_pack = mz.datasets.wiki_qa.load_data('train', task='ranking')
    valid_pack = mz.datasets.wiki_qa.load_data('dev', task='ranking')
    # Preprocess your input data in three lines of code, keep track parameters to be passed into the model.
    preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=40,
                                                      remove_stop_words=False)

    preprocessor = mz.preprocessors.DSSMPreprocessor()
    train_processed = preprocessor.fit_transform(train_pack)
    valid_processed = preprocessor.transform(valid_pack)
    return preprocessor, train_processed, valid_processed


def knrm_processed():
    train_pack_raw = mz.datasets.wiki_qa.load_data('train', task='ranking')
    dev_pack_raw = mz.datasets.wiki_qa.load_data('dev', task='ranking', filtered=True)
    test_pack_raw = mz.datasets.wiki_qa.load_data('test', task='ranking', filtered=True)
    preprocessor = mz.preprocessors.BasicPreprocessor(fixed_length_left=10, fixed_length_right=40,
                                                      remove_stop_words=False)

    train_pack_processed = preprocessor.fit_transform(train_pack_raw)
    valid_pack_processed = preprocessor.transform(dev_pack_raw)
    test_pack_processed = preprocessor.transform(test_pack_raw)
    return preprocessor, train_pack_processed, valid_pack_processed


def get_processed_data_from_cache():
    return load_from_pickle("matchzoo_prac1")


def tutorial():
    # data = get_processed_data()
    # data = knrm_processed()
    print("Loading data")
    data = get_processed_data_from_cache()
    preprocessor, train_processed, valid_processed = data
    # save_to_pickle(data, "matchzoo_prac1")
    print("Defining task")
    ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.MeanAveragePrecision()
    ]
    glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
    print('output_dim', glove_embedding.output_dim)

    #Initialize the model, fine-tune the hyper-parameters.
    print("building model")
    model = mz.models.KNRM()
    model.params.update(preprocessor.context)
    model.params['task'] = ranking_task
    model.params['embedding_output_dim'] = glove_embedding.output_dim
    model.params['embedding_trainable'] = True
    model.params['kernel_num'] = 21
    model.params['sigma'] = 0.1
    model.params['exact_sigma'] = 0.001
    model.params['optimizer'] = 'adadelta'
    model.build()
    model.compile()
    embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
    print(embedding_matrix.shape)
    # normalize the word embedding for fast histogram generating.
    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    print(l2_norm.shape)
    embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
    print(embedding_matrix.shape)
    model.load_embedding_matrix(embedding_matrix)

    print("defining generator")
    train_generator = mz.PairDataGenerator(train_processed, num_dup=1, num_neg=4, batch_size=64, shuffle=True)
    valid_x, valid_y = valid_processed.unpack()
    evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=len(valid_x))
    print("fitting")
    history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5,
                                  use_multiprocessing=False)


def tutorial_drmm():
    # data = get_processed_data()
    # data = knrm_processed()
    print("Loading data")
    data = get_processed_data_from_cache()
    preprocessor, train_processed, valid_processed = data
    # save_to_pickle(data, "matchzoo_prac1")
    print("Defining task")
    ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=4))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.MeanAveragePrecision()
    ]
    glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
    print('output_dim', glove_embedding.output_dim)

    #Initialize the model, fine-tune the hyper-parameters.
    print("building model")
    bin_size = 30
    model = mz.models.DRMM()
    model.params.update(preprocessor.context)
    model.params['input_shapes'] = [[10, ], [10, bin_size, ]]
    model.params['task'] = ranking_task
    model.params['mask_value'] = 0
    model.params['embedding_output_dim'] = glove_embedding.output_dim
    model.params['mlp_num_layers'] = 1
    model.params['mlp_num_units'] = 10
    model.params['mlp_num_fan_out'] = 1
    model.params['mlp_activation_func'] = 'tanh'
    model.params['optimizer'] = 'adadelta'
    model.build()
    model.compile()
    model.backend.summary()
    embedding_matrix = glove_embedding.build_matrix(preprocessor.context['vocab_unit'].state['term_index'])
    print(embedding_matrix.shape)
    # normalize the word embedding for fast histogram generating.
    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    print(l2_norm.shape)
    embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
    print(embedding_matrix.shape)
    model.load_embedding_matrix(embedding_matrix)
    hist_callback = mz.data_generator.callbacks.Histogram(embedding_matrix, bin_size=30, hist_mode='LCH')
    pred_generator = mz.DataGenerator(valid_processed, mode='point', callbacks=[hist_callback])
    pred_x, pred_y = pred_generator[:]
    print("defining generator")
    train_generator = mz.DataGenerator(train_processed, mode='pair', num_dup=5, num_neg=10, batch_size=20,
                                       callbacks=[hist_callback])

    valid_x, valid_y = valid_processed.unpack()
    evaluate = mz.callbacks.EvaluateAllMetrics(model, x=valid_x, y=valid_y, batch_size=len(valid_x))
    print("fitting")
    history = model.fit_generator(train_generator, epochs=20, callbacks=[evaluate], workers=5,
                                  use_multiprocessing=False)


if __name__ == "__main__":
    tutorial_drmm()